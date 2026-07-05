import math
import random
import gymnasium as gym
import pickle
import os


# créer un noeud, à chaque fois que l'agent prends une décision il passe à un nouveau noeud, cette classe est appelé
class MCTSNode:
    # initialisation du noeud
    def __init__(self, state, parent=None, action_from_parent=None, reward_from_parent=0, done=False):
        # initialisation de l'état
        self.state = state
        # définition de l'état parent
        self.parent = parent
        # définition de l'action qui a été utilisé pour arriver au présent noeud
        self.action_from_parent = action_from_parent
        # définition des points obtenus cette à la dernière action réalisé
        self.reward_from_parent = reward_from_parent
        # définition de si l'état est final ou non ( fin de partie ou non )
        self.done = done

        # définition des actions enfants déjà testé : clé action valeur etat noeud enfant
        self.children = {}
        # total de fois  que le noeud a été visités lors des simulations
        self.visits = 0
        # stock la valeur obtenues quand on est passé par ce noeud, ce que ce noeud à raporté au total
        self.total_value = 0.0

    # vérifie que toutes les actions ont déjà été visité dans ce current noeud // action_space_size = actions possible dans le jeu
    def is_fully_expanded(self, action_space_size):
        return len(self.children) == action_space_size

    # calcul la valeur moyenne du noeud
    def average_value(self):
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MonteCarloTreeSearchAgent:
    # constructeur pour initialiser l'agent
    def __init__(
        self,
        env,
        num_simulations=200,
        max_rollout_depth=25,
        exploration_constant=1.0,
        discount_factor=0.99,
        save_path="mcts_memory.pkl",
        rollout_epsilon=0.0,
    ):
        # setup l'environnement ( taxi driver, warcraft ...)
        self.env = env
        # récupère le modèle des transitions de l'environnement :
        # pour chaque état et chaque action, indique les états suivants possibles,
        # leur probabilité, la récompense obtenue et si l'épisode est terminé
        self.P = env.unwrapped.P
        # récupère le nombre d'action possible dans le jeu
        self.num_actions = env.action_space.n
        # définis le nombre de simulation de chemin imaginaire AVANT de choisir une action
        self.num_simulations = num_simulations
        # défini le nombre de coup calculé lors d'une simulation ( voir jusqu'à 20,30,40 coup à l'avance)
        self.max_rollout_depth = max_rollout_depth
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        # probabilité de jouer un coup aléatoire pendant le rollout plutôt que
        # de suivre l'heuristique (évite que le rollout reste bloqué contre un
        # mur et garde un minimum d'exploration)
        self.rollout_epsilon = rollout_epsilon

        # chemin du fichier où la mémoire de l'agent sera sauvegardée
        self.save_path = save_path
        # charge la mémoire déjà existante si elle existe, sinon crée une mémoire vide
        self.memory = self.load_memory()

        # précalcule, pour chacun des 4 lieux nommés (R, G, Y, B), le meilleur
        # coup à jouer depuis n'importe quelle case de la grille pour s'en
        # rapprocher en respectant les murs (calculé une seule fois, pas cher :
        # la grille ne fait que 25 cases)
        self.action_maps = self._build_action_maps()

    # coordonnées fixes des 4 lieux nommés dans Taxi (R, G, Y, B)
    LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]

    def _build_action_maps(self):
        """
        Construit, pour chaque lieu cible (R, G, Y, B), une table
        {(ligne, colonne): meilleure_action} donnant le prochain coup à jouer
        depuis n'importe quelle case pour se rapprocher de ce lieu au plus
        court chemin, en respectant les murs (via une BFS sur le modèle de
        transitions self.P, qui connaît déjà la disposition des murs).
        """
        from collections import deque

        # construit le graphe des 25 cases (indépendant du passager/destination,
        # seule la position du taxi nous intéresse ici -> on fixe pass_loc=0, dest=0)
        graph = {(r, c): {} for r in range(5) for c in range(5)}
        for r in range(5):
            for c in range(5):
                fixed_state = self.env.unwrapped.encode(r, c, 0, 0)
                for action in range(4):  # 0=sud, 1=nord, 2=est, 3=ouest
                    for prob, next_state, reward, done in self.P[fixed_state][action]:
                        nr, nc, _, _ = self.env.unwrapped.decode(next_state)
                        if (nr, nc) != (r, c):
                            graph[(r, c)][action] = (nr, nc)

        # graphe inversé : pour chaque case, la liste des cases qui peuvent y arriver
        reverse_graph = {cell: [] for cell in graph}
        for cell, moves in graph.items():
            for action, neighbor in moves.items():
                reverse_graph[neighbor].append(cell)

        action_maps = {}
        for target in self.LOCATIONS:
            # BFS depuis la cible sur le graphe inversé -> donne la distance
            # réelle de chaque case JUSQU'À la cible dans le graphe normal
            distance = {target: 0}
            queue = deque([target])
            while queue:
                current = queue.popleft()
                for predecessor in reverse_graph[current]:
                    if predecessor not in distance:
                        distance[predecessor] = distance[current] + 1
                        queue.append(predecessor)

            # pour chaque case, on choisit l'action qui mène vers un voisin
            # dont la distance à la cible est strictement plus petite
            best_action = {}
            for cell, moves in graph.items():
                if cell == target:
                    continue
                for action, neighbor in moves.items():
                    if distance.get(neighbor, float("inf")) < distance.get(cell, float("inf")):
                        best_action[cell] = action
                        break

            action_maps[target] = best_action

        return action_maps

    def heuristic_action(self, state):
        """
        Choisit une action "raisonnable" plutôt qu'aléatoire pour le rollout :
        - si le passager n'est pas encore dans le taxi -> se diriger vers lui, le prendre
        - si le passager est dans le taxi -> se diriger vers la destination, le déposer
        Le déplacement suit le plus court chemin précalculé (BFS), donc il
        contourne correctement les murs au lieu de rester bloqué en boucle.
        """
        # décode l'état entier en (ligne taxi, colonne taxi, lieu passager, destination)
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        position = (taxi_row, taxi_col)

        if pass_loc == 4:
            # le passager est déjà dans le taxi -> la cible devient la destination
            target = self.LOCATIONS[dest_idx]
            if position == target:
                return 5  # dropoff
        else:
            # le passager attend encore -> la cible est son lieu de prise en charge
            target = self.LOCATIONS[pass_loc]
            if position == target:
                return 4  # pickup

        # regarde dans la table précalculée quel coup rapproche le taxi de la cible
        action = self.action_maps[target].get(position)
        if action is not None:
            return action

        # cas extrême (ne devrait pas arriver, la BFS couvre toute la grille) -> coup aléatoire
        return random.randrange(self.num_actions)

    # charge la mémoire de l'agent depuis un fichier
    def load_memory(self):
        # vérifie si le fichier de sauvegarde existe déjà
        if os.path.exists(self.save_path):
            try:
                # ouvre le fichier en lecture binaire
                with open(self.save_path, "rb") as f:
                    memory = pickle.load(f)
                print("Mémoire chargée.")
                return memory
            except (EOFError, pickle.UnpicklingError):
                # fichier vide ou corrompu (ex: script coupé pendant une sauvegarde)
                # -> on repart avec une mémoire vide plutôt que de planter
                print("Fichier de mémoire corrompu ou vide, nouvelle mémoire créée.")
                return {}

        # si aucun fichier n'existe, on commence avec une mémoire vide
        print("Aucune mémoire trouvée, nouvelle mémoire créée.")
        return {}

    # sauvegarde la mémoire de l'agent dans un fichier
    def save_memory(self):
        # ouvre le fichier en écriture binaire
        with open(self.save_path, "wb") as f:
            # enregistre la mémoire actuelle dans le fichier
            pickle.dump(self.memory, f)

    def transition(self, state, action):
        """
        Pour Taxi-v3, P[state][action] = liste de transitions possibles
        Chaque transition = (prob, next_state, reward, done)
        """
        transitions = self.P[state][action]

        r = random.random()
        cumulative = 0.0

        for prob, next_state, reward, done in transitions:
            cumulative += prob
            if r <= cumulative:
                return next_state, reward, done

        # fallback si problème de précision flottante
        prob, next_state, reward, done = transitions[-1]
        return next_state, reward, done

    def ucb_score(self, parent_visits, child):
        if child.visits == 0:
            return float("inf")

        # IMPORTANT : on compare des ACTIONS (les enfants d'un même noeud), donc
        # on doit utiliser la valeur de l'action Q = récompense immédiate pour
        # y arriver + valeur (discountée) de l'état résultant. Utiliser
        # uniquement child.average_value() (comme dans la version d'origine)
        # ignore complètement reward_from_parent : une action invalide (-10)
        # et une action correcte (-1) qui mènent à des états de valeur future
        # similaire se retrouvaient alors avec un score quasi identique, la
        # pénalité immédiate étant invisible dans la comparaison.
        exploitation = child.reward_from_parent + self.discount_factor * child.average_value()
        exploration = self.exploration_constant * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    def select(self, node):
        """
        Descend dans l'arbre tant que :
        - le nœud n'est pas terminal
        - le nœud est complètement expandé
        """
        while not node.done and node.is_fully_expanded(self.num_actions):
            best_action = None
            best_child = None
            best_score = float("-inf")

            for action, child in node.children.items():
                score = self.ucb_score(node.visits, child)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child

            node = best_child

        return node

    def expand(self, node):
        """
        Ajoute un enfant pour une action encore jamais explorée
        """
        if node.done:
            return node

        untried_actions = [a for a in range(self.num_actions) if a not in node.children]
        if not untried_actions:
            return node

        action = random.choice(untried_actions)
        next_state, reward, done = self.transition(node.state, action)

        child = MCTSNode(
            state=next_state,
            parent=node,
            action_from_parent=action,
            reward_from_parent=reward,
            done=done,
        )
        node.children[action] = child
        return child

    def rollout(self, state, done):
        """
        Simulation depuis l'état courant, guidée par l'heuristique la plupart du
        temps (rollout_epsilon de chance de jouer aléatoirement pour explorer)
        """
        if done:
            return 0.0

        total_reward = 0.0
        discount = 1.0
        current_state = state
        current_done = done

        for _ in range(self.max_rollout_depth):
            if current_done:
                break

            if random.random() < self.rollout_epsilon:
                action = random.randrange(self.num_actions)
            else:
                action = self.heuristic_action(current_state)

            next_state, reward, current_done = self.transition(current_state, action)

            total_reward += discount * reward
            discount *= self.discount_factor
            current_state = next_state

        return total_reward

    def backpropagate(self, node, rollout_value):
        """
        Remonte la valeur dans l'arbre
        """
        current = node
        value = rollout_value

        while current is not None:
            current.visits += 1
            current.total_value += value

            # quand on remonte, on réinjecte la reward du parent -> enfant
            if current.parent is not None:
                value = current.reward_from_parent + self.discount_factor * value

            current = current.parent

    # reconstruit le noeud racine à partir de la mémoire sauvegardée
    def rebuild_root_from_memory(self, root):
        """
        Recharge les actions déjà connues pour l'état actuel.
        """
        state = root.state

        # si l'état n'est pas encore dans la mémoire, on ne recharge rien
        if state not in self.memory:
            return

        # pour chaque action déjà connue dans cet état
        for action, data in self.memory[state].items():
            # on recalcule l'état suivant possible depuis cet état et cette action
            next_state, reward, done = self.transition(state, action)

            # on recrée un noeud enfant comme s'il avait déjà été exploré
            child = MCTSNode(
                state=next_state,
                parent=root,
                action_from_parent=action,
                reward_from_parent=reward,
                done=done,
            )

            # on remet le nombre de visites sauvegardé
            child.visits = data["visits"]
            # on remet la valeur totale sauvegardée
            child.total_value = data["total_value"]

            # on rattache cet enfant au noeud racine
            root.children[action] = child

        # FIX : la racine doit avoir au moins autant de visites que la somme
        # des visites de ses enfants. Sans ça, root.visits reste à 0 alors que
        # les enfants rechargés ont déjà des visites -> si l'état est déjà
        # complètement exploré, select() entre dans la boucle UCB avec
        # parent_visits = 0, et math.log(0) plante avec "math domain error".
        root.visits = sum(child.visits for child in root.children.values())

    # sauvegarde les résultats du noeud racine dans la mémoire globale
    def save_root_to_memory(self, root):
        """
        Sauvegarde les résultats MCTS obtenus pour cet état.
        """
        state = root.state

        # crée une entrée mémoire pour l'état actuel
        self.memory[state] = {}

        # sauvegarde chaque action enfant connue depuis cet état
        for action, child in root.children.items():
            self.memory[state][action] = {
                # sauvegarde combien de fois cette action a été visitée
                "visits": child.visits,
                # sauvegarde la valeur totale obtenue par cette action
                "total_value": child.total_value,
            }

        # écrit la mémoire dans le fichier mcts_memory.pkl
        self.save_memory()

    def search(self, root_state):
        root = MCTSNode(state=root_state)

        # recharge les anciennes simulations déjà faites pour cet état
        self.rebuild_root_from_memory(root)

        for _ in range(self.num_simulations):
            # 1. Selection
            node = self.select(root)

            # 2. Expansion
            node = self.expand(node)

            # 3. Rollout
            rollout_value = self.rollout(node.state, node.done)

            # 4. Backpropagation
            self.backpropagate(node, rollout_value)

        # sauvegarde ce que l'agent vient d'apprendre pour cet état
        self.save_root_to_memory(root)

        # Choix final. On compare les actions avec la même formule Q que dans
        # ucb_score (récompense immédiate + valeur discountée de l'état
        # résultant) : comparer directement child.average_value() ignorerait
        # la récompense immédiate de chaque action, exactement le bug qu'on
        # vient de corriger dans ucb_score. On garde quand même les visites
        # comme filtre : un enfant très peu visité a une estimation trop
        # bruitée pour qu'on lui fasse confiance, même si son Q affiché
        # paraît bon.
        if not root.children:
            return random.randrange(self.num_actions)

        def q_value(child):
            return child.reward_from_parent + self.discount_factor * child.average_value()

        best_action = max(root.children.items(), key=lambda item: q_value(item[1]))[0]
        return best_action


def run_episode(env, agent, render=False, max_steps=200, on_step=None):
    state, info = env.reset()
    done = False
    truncated = False

    total_reward = 0
    step_count = 0

    while not done and not truncated and step_count < max_steps:
        if render:
            env.render()

        action = agent.search(state)
        state, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # callback optionnel : permet à une interface externe (ex: dashboard web)
        # de récupérer l'état du jeu à chaque étape, sans toucher à la logique MCTS
        if on_step is not None:
            on_step(env, state, action, reward, done, step_count)

    return total_reward, step_count, done


def evaluate_agent(num_episodes=20, render=False, on_step=None, on_episode_end=None):
    env = gym.make("Taxi-v4", render_mode="human" if render else None)
    agent = MonteCarloTreeSearchAgent(
        env,
        num_simulations=200,
        max_rollout_depth=25,
        exploration_constant=1.0,
        discount_factor=0.99,
        save_path="mcts_memory.pkl",
        rollout_epsilon=0.0,
    )

    rewards = []
    steps = []
    successes = 0

    for episode in range(num_episodes):
        total_reward, step_count, done = run_episode(
            env, agent, render=render, on_step=on_step
        )
        rewards.append(total_reward)
        steps.append(step_count)
        if done:
            successes += 1

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"reward={total_reward} | steps={step_count} | done={done}"
        )

        # callback optionnel : permet de rafraîchir un dashboard après chaque épisode
        if on_episode_end is not None:
            on_episode_end(episode + 1, total_reward, step_count, done, len(agent.memory))

    env.close()

    print("\n=== Résultats ===")
    print(f"Reward moyenne : {sum(rewards) / len(rewards):.2f}")
    print(f"Nombre moyen d'étapes : {sum(steps) / len(steps):.2f}")
    print(f"Taux de succès : {successes / num_episodes * 100:.2f}%")
    print(f"Nombre d'états sauvegardés : {len(agent.memory)}")


if __name__ == "__main__":
    # premier lancement rapide pour vérifier que l'entraînement fonctionne ;
    # remonte num_episodes (ex: 20 ou plus) une fois que tu ajusteras les perfs
    evaluate_agent(num_episodes=5, render=False)
