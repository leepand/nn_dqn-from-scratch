import network as nn
import pickle
import numpy as np
import random


def argmax_rand(dict_arr):
    """Return key with maximum value, break ties randomly."""
    assert isinstance(dict_arr, dict)
    # Find the maximum value in the dictionary
    max_value = max(dict_arr.values())
    # Get a list of keys with the maximum value
    max_keys = [key for key, value in dict_arr.items() if value == max_value]
    # Randomly select one key from the list
    selected_key = random.choice(max_keys)
    # Return the selected key
    return selected_key


class trainer_config:
    """
    configuration for the Q learner (trainer) for easy reuse
    everything not model related goes here. maybe
    """

    def __init__(
        self,
        model_name,
        actions,
        input_dim,
        BUFFER_SIZE=50e3,
        STEPS_PER_EPISODE=500,
        MAX_STEPS=100000,
        UPDATE_TARGET_STEPS=1000,
        BATCH_SIZE=32,
        GAMMA=0.99,
        EXPLORATION=100,
        E_MIN=0.01,
        priority=False,
        alpha=0.6,
        epsilon=0.4,
    ):
        ### game environment
        self.model_name = model_name
        ### world variables for model buildings
        self.INPUT_SIZE = input_dim
        self.OUTPUT_SIZE = len(actions)
        ### training variables
        self.BUFFER_SIZE = BUFFER_SIZE
        self.STEPS_PER_EPISODE = STEPS_PER_EPISODE
        self.MAX_STEPS = MAX_STEPS
        self.UPDATE_TARGET_STEPS = UPDATE_TARGET_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EXPLORATION = EXPLORATION
        self.E_MIN = E_MIN
        #### PRIO MODULE ( default := alpha= 0.,epsilon=0.01)
        self.priority = priority
        self.alpha = alpha
        self.epsilon = epsilon


class DQNAgent:
    def __init__(
        self, actions=[], input_dim=3, hidden_dim=32, model_db=None, no_bias=False
    ):
        self._model_db = model_db
        self.actions = actions
        self.epsilon_decay = 0.01
        # STEP 1: create configuration
        self.config = trainer_config(
            model_name="model-v0",
            actions=actions,
            input_dim=input_dim,
            MAX_STEPS=100000,
        )
        self.no_bias = no_bias
        # STEP 1: create configuration
        A1 = nn.layer(self.config.INPUT_SIZE, hidden_dim, no_bias)
        # A2 = nn.layer(128, 64)
        AOUT = nn.layer(hidden_dim, self.config.OUTPUT_SIZE, no_bias)
        AOUT.f = nn.f_iden
        L1 = nn.layer(self.config.INPUT_SIZE, hidden_dim, no_bias)
        # L2 = nn.layer(128, 64)
        LOUT = nn.layer(hidden_dim, self.config.OUTPUT_SIZE, no_bias)
        LOUT.f = nn.f_iden
        self.onlineNet = nn.mlp([A1, AOUT], no_bias=self.no_bias)
        self.targetNet = nn.mlp([L1, LOUT], no_bias=self.no_bias)
        self.model_params = {
            "w0": A1.w,
            "w1": AOUT.w,
            "m10": A1.m1,
            "m20": A1.m2,
            "m11": AOUT.m1,
            "m21": AOUT.m2,
        }
        self.hidden_dim = hidden_dim
        self.eps = self.config.epsilon

    def _init_model(self):
        params = {}
        Layerlist = []
        i = 0
        l0 = nn.layer(self.config.INPUT_SIZE, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.config.OUTPUT_SIZE, self.no_bias)
        Layerlist = [l0, l1]
        for L in Layerlist:
            params[f"w{i}"] = L.w
            params[f"m1{i}"] = L.m1
            params[f"m2{i}"] = L.m2
            # print(L.w,"w",L.w.shape,L.m1,L.m2,"m")
            i += 1
        params["model_updated_cnt"] = 0
        return params

    def _init_weights(self):
        weights = {}
        Layerlist = []
        i = 0
        l0 = nn.layer(self.config.INPUT_SIZE, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.config.OUTPUT_SIZE, self.no_bias)
        Layerlist = [l0, l1]
        for L in Layerlist:
            weights[f"w{i}"] = L.w
            i += 1
        weights["model_updated_cnt"] = 0
        return weights

    def get_model(self, model_id):
        model_key = f"{model_id}:model"
        model = self._model_db.get(model_key)
        if model is None:
            model = self._init_model()
        else:
            model = pickle.loads(model)
        return model

    def get_weights(self, model_id):
        model_key = f"{model_id}:params"
        model = self._model_db.get(model_key)
        if model is None:
            model = self._init_weights()
        else:
            model = pickle.loads(model)
        return model

    def save_model(self, model, model_id):
        model_key = f"{model_id}:model"
        model["model_updated_cnt"] += 1
        self._model_db.set(model_key, pickle.dumps(model))

    def save_weights(self, model, model_id):
        model_key = f"{model_id}:params"
        model["model_updated_cnt"] += 1
        self._model_db.set(model_key, pickle.dumps(model))

    def act(self, state, model_id, allowed=None, not_allowed=None):
        model_weights = self.get_weights(model_id=model_id)
        model_updated_cnt = model_weights.get("model_updated_cnt", 1)
        epsilon = max(self.eps - self.epsilon_decay * model_updated_cnt, 0.1)
        if allowed is None:
            valid_actions = self.actions
        else:
            valid_actions = allowed
        if not_allowed is not None:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)
        epsilon = 0.2
        if random.random() < epsilon:

            action = random.choice(valid_actions)
        else:
            action_probs = self.onlineNet.infer(
                input_=state, w_s=model_weights, predict=True
            )
            action_probs_dict = {
                a: action_probs[range(action_probs.shape[0]), a] for a in valid_actions
            }
            action = argmax_rand(action_probs_dict)
        return action

    def learn(self, state, action, next_state, reward, model_id, done=False):
        """updates the onlineDQN with target Q values for
        the greedy action(choosen by onlineDQN)
        """
        model = self.get_model(model_id=model_id)
        Q_old, h_list, x_list = self.onlineNet.infer(state, w_s=model, predict=False)
        Q = np.copy(Q_old)
        t = self.onlineNet.infer(next_state, w_s=model, predict=True)
        a = np.argmax(t, axis=1)
        Q[range(Q.shape[0]), int(action)] = (
            reward + np.logical_not(done) * self.config.GAMMA * t[range(t.shape[0]), a]
        )
        model, weights = self.onlineNet.train(
            Q_=Q_old, input_=x_list, target_=Q, h1=h_list, model=model,x=state
        )
        self.save_model(model=model, model_id=model_id)
        self.save_weights(model=weights, model_id=model_id)

    def argmax_rand(self, a):
        """Break ties randomly, (np.argmax always picks first max)."""
        assert isinstance(a, list) or a.ndim == 1
        return np.random.choice(np.flatnonzero(a == np.max(a)))

    def _get_valid_actions(self, forbidden_actions, all_actions=None):
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if all_actions is None:
            all_actions = self.actions
        if forbidden_actions is None:
            forbidden_actions = set()
        else:
            forbidden_actions = set(forbidden_actions)

        if not all(a in all_actions for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(all_actions) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError(
                "All actions are forbidden. You must allow at least 1 action."
            )

        valid_actions = list(valid_actions)
        return valid_actions
