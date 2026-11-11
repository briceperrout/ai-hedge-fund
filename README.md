# AI Hedge Fund — étude d'une architecture neuro-symbolique pour la décision sous risque

Un modèle statistique propose des trades, un comité de LLM locaux a le droit de les bloquer,
et du code Python déterministe garde seul la main sur l'argent.

Ce dépôt est un **projet de recherche et d'apprentissage**. Il ne passe aucun ordre réel, ne promet
aucune performance, et sa conclusion provisoire est honnête : testé sur huit ans, le signal ne bat pas le
marché et ne choisit pas mieux que le hasard. L'intérêt est dans la méthode — comment construire un tel système sans qu'il triche,
et comment mesurer ce que les LLM y apportent vraiment.

## La question posée

Un LLM qui lit l'actualité peut-il améliorer un signal quantitatif en lui opposant un veto ?
Et surtout : comment le savoir, sachant que bloquer des trades au hasard change aussi les résultats ?

## Architecture

```mermaid
flowchart LR
    subgraph G["Cerveau gauche — statistique"]
        P[(Prix, VIX, taux<br/>SQLite)] --> X[XGBoost<br/>probabilité que l'objectif<br/>soit atteint avant le stop]
        O[Optuna<br/>walk-forward] -.hyperparamètres.-> X
    end
    subgraph M["Mémoire — Qdrant"]
        N[Dépêches RSS<br/>dépôts SEC]
        J[Journal d'expérience<br/>leçons des pertes]
    end
    subgraph D["Cerveau droit — LangGraph"]
        B1[Bull<br/>Llama 3.1 8B] --> C[CRO<br/>Qwen 2.5 32B]
        B2[Bear<br/>Llama 3.1 8B] --> C
    end
    X -- "p ≥ 70 %" --> DOS[Dossier<br/>d'investissement]
    N --> DOS
    J --> DOS
    DOS --> B1
    DOS --> B2
    C -- "JSON validé par Pydantic<br/>autoriser ou bloquer" --> K[Python<br/>Kelly fractionné<br/>stop-loss, objectif]
    K --> E[Exécution<br/>simulateur ou Alpaca paper]
    E -- "trade perdant" --> A[Autopsie<br/>nocturne]
    A --> J
```

Trois règles tiennent l'ensemble :

1. **Le LLM ne peut que bloquer.** Sa sortie est un objet `VetoDecision` strict (`extra="forbid"`,
   validateurs de cohérence). Il ne choisit ni la valeur, ni la taille, ni le stop. Il ne peut pas
   créer un ordre que XGBoost n'a pas émis.
2. **En cas de doute, on ne trade pas.** Modèle éteint, JSON invalide après trois relances, délai
   dépassé : le verdict devient un veto de sécurité.
3. **Rien ne regarde le futur.** Features, cibles, dépêches et leçons sont filtrées par date ; le
   modèle n'est entraîné que sur des trades dont l'issue était déjà connue.

Tout tourne en local : Ollama et LM Studio pour les modèles, Qdrant pour les vecteurs, SQLite pour les prix.

## Résultats

### Moteur quantitatif seul, année 2023 (hors échantillon)

| | Stratégie | S&P 500 (achat-conservation) |
|---|---|---|
| Rendement | +1,73 % | +26,7 % |
| Ratio de Sharpe | 1,29 | — |
| Perte maximale | −1,29 % | — |
| Trades | 24, dont 75 % gagnants | — |

Lecture : la stratégie est investie une petite fraction du temps (Kelly au quart, dix valeurs), d'où
une courbe presque plate. Le Sharpe est flatteur mais repose sur 24 trades ; l'AUC du classifieur en
validation est proche de 0,5. **Le test élargi ci-dessous montre que ce résultat ne tient pas.** Les
hyperparamètres ont été choisis par Optuna sur des données antérieures à 2023.

Les chiffres varient légèrement d'un jour à l'autre (un trade de plus ou de moins) parce que Yahoo
recalcule les prix ajustés ; à données identiques, le backtest est déterministe.

### Robustesse : huit ans, cinquante valeurs, et une référence aléatoire

L'année 2023 était une fenêtre favorable. `run_robustness.py` rejoue le moteur quantitatif de 2019 à
novembre 2026, sur l'univers d'origine puis sur quarante grandes valeurs qu'Optuna n'a jamais vues.

| | Univers d'origine (10) | Valeurs jamais vues (40) | S&P 500 |
|---|---|---|---|
| Rendement sur la période | +2,4 % | +15,5 % | +236 % |
| Ratio de Sharpe | 0,08 | 0,27 | — |
| Perte maximale | −21,1 % | −20,2 % | — |
| Trades | 339 | 247 | — |

Un taux de réussite élevé ne prouve rien ici : avec un objectif à 1,5 ATR et un stop à 3 ATR, une
entrée au hasard gagne déjà une fois sur deux. La vraie question est de savoir si les signaux font
mieux que le hasard, et pourquoi.

| | Univers d'origine | Valeurs jamais vues |
|---|---|---|
| Rendement moyen par signal | +0,77 % | +1,89 % |
| Rendement moyen d'une entrée au hasard, mêmes années | +0,98 % | +0,59 % |
| Jours à signal : rendement moyen de tout l'univers | +0,80 % | +0,99 % |
| Autres jours : rendement moyen de tout l'univers | +0,66 % | +0,42 % |
| Écart entre la valeur désignée et la moyenne du jour | +0,06 % (t = 0,19) | −0,17 % (t = −0,67) |

**Conclusion : le modèle ne sait pas choisir une action.** Les jours où il émet un signal, la valeur
qu'il désigne ne fait pas mieux que la moyenne de l'univers. L'écart apparent sur les quarante valeurs
vient du calendrier : les signaux se concentrent sur les périodes de forte volatilité (984 en 2020,
495 en 2022), où tout acheter après une chute rapportait davantage. C'est un effet « acheter le creux »,
porté par un petit nombre d'épisodes de marché, pas une capacité de sélection.

Les t de Student sont calculés sur des blocs de dix séances, parce que les trades se chevauchent.
Biais restants : univers choisi aujourd'hui, exécution idéalisée.

### Valeur du veto : l'expérience contrôlée

`run_experiment.py` rejoue la même période dans trois configurations qui partagent exactement les
mêmes modèles XGBoost, donc les mêmes signaux :

| Bras | Politique |
|---|---|
| A | tout signal est exécuté |
| B | débat Bull/Bear puis veto du CRO |
| C | veto tiré au hasard, au même taux que le CRO, répété N fois |

Le bras C est le témoin. Sur 2023, un veto aléatoire à 30 % fait *baisser* le Sharpe (médiane 0,74
sur 30 tirages, intervalle 5 %–95 % de 0,17 à 1,34, contre 1,29 sans veto) : filtrer n'est pas
gratuit. Le CRO n'a de valeur que s'il sort du haut de cette distribution, et si les signaux qu'il
bloque auraient réellement moins rapporté que ceux qu'il autorise (test de permutation).

**État actuel : le bras B n'a pas encore été exécuté avec les modèles prévus** (Llama 3.1 8B et
Qwen 2.5 32B demandent plus de mémoire que la machine de développement). Un essai technique de dix
jours avec un petit modèle a validé la chaîne complète — 5 signaux, 0 JSON rejeté, 1 autopsie — mais
ne dit rien de la qualité des verdicts. Le script refuse de conclure en dessous de cinq signaux par groupe.

### Une réserve qui ne se corrige pas en backtest

Un LLM entraîné après 2023 « connaît » 2023. Lui faire juger un trade de mars 2023 n'est pas un test
propre, même si le dossier qu'on lui fournit est filtré par date. La seule mesure sans ce biais est le
**paper-trading en temps réel** : le tableau de bord recalcule, pour chaque signal bloqué, ce que le
trade serait devenu une fois son issue connue.

### Le LLM comme source de variables : lecture des dépôts SEC

Dans le comité, le LLM ne fait que filtrer. Une seconde piste lui donne un rôle plus défendable : lire en
entier les 8-K, 10-Q et 10-K et en tirer des variables chiffrées que XGBoost peut exploiter.

```mermaid
flowchart LR
    S[SEC EDGAR<br/>8-K + annexes, 10-Q, 10-K] --> T[Texte daté<br/>à la minute de publication]
    T --> N[Notes par tranche<br/>petit modèle]
    N --> V[FilingSignals<br/>validé par Pydantic]
    V --> F[Features jour par jour<br/>sans fuite temporelle]
    F --> X[XGBoost]
    V -. contrôle .-> Q[XBRL officiel<br/>citations retrouvées]
```

Le document est découpé, un petit modèle prend des notes sur chaque tranche, puis le modèle d'arbitrage les
convertit en onze variables : chiffre d'affaires et BPA du trimestre, croissance, prévisions, ton sur la
demande, marges, risques de liquidité et juridique, retour aux actionnaires. Un communiqué publié à 16 h 30
n'alimente le modèle qu'à partir du lendemain.

`run_filings_study.py` évalue trois choses, de la plus vérifiable à la plus incertaine :

1. **Coût** : appels, jetons et secondes par document.
2. **Exactitude** : chiffre d'affaires et BPA comparés aux données XBRL de la SEC ; citations recherchées
   mot pour mot dans le document ; nature du document comparée aux rubriques officielles du 8-K.
3. **Utilité** : gain d'AUC hors échantillon, comparé à un témoin où chaque société reçoit les variables
   d'une autre. Si le gain survit à l'échange, il ne venait pas du contenu des dépôts.

**État actuel : essai technique sur six dépôts d'Apple, avec un modèle de 3 milliards de paramètres**
(les modèles cibles ne tiennent pas sur la machine de développement) :

| Mesure | Résultat |
|---|---|
| Temps de lecture | 3 minutes par document en moyenne, 18 minutes au total |
| Chiffre d'affaires trimestriel | 3 exacts sur 4, 1 erreur d'unité |
| BPA dilué | 2 exacts sur 4, 1 faux, 1 omis |
| Citations retrouvées dans le texte | 75 % |
| Communiqués de résultats reconnus comme tels | 0 sur 2 |
| Utilité prédictive | non concluant : il faut au moins 100 dépôts lus |

Ces chiffres mesurent un petit modèle sur un échantillon minuscule : ils montrent surtout que l'évaluation
attrape les erreurs (unité, omission, mauvaise étiquette). Les variables ne sont pas branchées sur la
stratégie tant que l'étude n'a pas tourné à grande échelle (`ENABLE_FILING_FEATURES=false`).

## Ce que les tests ont attrapé

Quatre défauts trouvés en exécutant réellement le système, typiques de ce genre de projet :

- **Un P/E faux d'un facteur dix** sur NVDA : les prix historiques sont ajustés des divisions
  d'actions, pas les bénéfices par action publiés à l'époque. Corrigé par une table des splits.
- **Un plantage silencieux** quand XGBoost s'entraîne après le chargement de PyTorch sur macOS (deux
  runtimes OpenMP dans le même processus).
- **« Event loop is closed » au deuxième signal** : `asyncio.run` fermait la boucle à laquelle les
  clients HTTP des modèles restaient attachés.
- **Des relances qui n'avaient jamais lieu** : quand un validateur Pydantic rejetait la réponse du
  modèle, le SDK levait l'erreur avant le code de relance. Le comportement restait sûr (veto par
  défaut), mais chaque réponse imparfaite était perdue au lieu d'être corrigée.

La suite automatisée (134 tests, trois secondes, sans réseau ni LLM) couvre le verrou Pydantic, le
critère de Kelly, l'absence de fuite temporelle, le moteur d'exécution, la chaîne de décision, les
statistiques de l'expérience, le tableau de bord et la lecture des dépôts SEC. Les tests anti-fuite ont été vérifiés par
mutation : une moyenne mobile centrée ou une purge retirée les font échouer.

## Installation

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Pour le comité IA : [Ollama](https://ollama.com) avec `llama3.1:8b`, et [LM Studio](https://lmstudio.ai)
avec un Qwen 2.5 32B servi en mode serveur. Qdrant est optionnel : sans serveur, une base embarquée
est créée dans `data/qdrant_local`.

## Utilisation

```bash
python -m pytest                                  # tests
python -m data_pipeline.fetch_history             # télécharge prix et indicateurs macro
python -m quant_engine.optimizer                  # recherche d'hyperparamètres (Optuna)
python run_backtest.py --no-llm                   # moteur quantitatif seul, 2023
python run_backtest.py                            # système complet (modèles requis)
python run_experiment.py                          # les trois bras + verdict statistique
python run_experiment.py --veto-rate 0.3          # sans LLM : bras A et C seulement
python run_robustness.py                          # 2019 à aujourd'hui, deux univers, référence aléatoire
python dashboard.py --run <identifiant> --open    # page de synthèse d'un backtest
```

Lecture des dépôts SEC (les deux premières commandes sont longues et reprennent où elles s'étaient arrêtées) :

```bash
python -m data_pipeline.fetch_filings             # texte intégral des 8-K, 10-Q, 10-K
python -m data_pipeline.fetch_news                # chiffres XBRL de référence
python -m llm_agents.filing_extractor             # lecture par les LLM (modèles requis)
python run_filings_study.py                       # coût, exactitude, utilité
```

### Paper-trading en continu

```bash
python run_live.py --loop --at 16:30    # un cycle par jour ouvré, après la clôture de New York
python run_live.py --quant-only         # machine sans modèles : portefeuille témoin seul
python run_live.py --status             # état des portefeuilles
python dashboard.py --open              # tableau de bord (data/dashboard.html)
```

Deux portefeuilles simulés vivent côte à côte sur les mêmes signaux : l'un soumis au veto du CRO,
l'autre qui exécute tout (témoin). Leur écart, mesuré en temps réel, est la seule évaluation du comité
qui échappe au biais de contamination. Chaque cycle met à jour les données, gère les positions, soumet
les nouveaux signaux au comité, autopsie les pertes, puis régénère `data/dashboard.html`. Pour que la boucle survive à la fermeture
du terminal : `tmux`, ou `nohup python run_live.py --loop > live.log 2>&1 &`.

Le broker par défaut est un simulateur local qui partage son moteur d'exécution avec le backtest.
Un connecteur Alpaca (compte papier uniquement) existe mais **n'a pas été testé**.

## Organisation du code

```
data_pipeline/   prix et macro (SQLite), dépêches RSS et dépôts SEC, Qdrant, dossier d'investissement
quant_engine/    features, cibles par barrières, XGBoost, Kelly, optimisation Optuna
llm_agents/      schémas Pydantic, prompts, graphe LangGraph (comité et autopsie)
run_backtest.py  simulation walk-forward et moteur d'exécution
run_experiment.py  expérience contrôlée sur la valeur du veto
run_filings_study.py  évaluation des variables extraites des dépôts SEC
run_robustness.py  robustesse du signal dans le temps et sur d'autres valeurs
run_live.py      cycle quotidien de paper-trading
dashboard.py     tableau de bord HTML
tests/           suite pytest
```

## Limites connues

- **Pas d'avantage de sélection** : testé sur huit ans et cinquante valeurs, le signal ne désigne pas de
  meilleures actions que le hasard (voir Robustesse).
- **Biais du survivant** : l'univers est choisi aujourd'hui, avec des valeurs qui ont bien tourné.
- **Dépêches historiques incomplètes** : la recherche datée de Google News ne restitue pas ce qui
  était réellement lisible le jour même.
- **Exécution idéalisée** : remplissage à l'ouverture, coûts forfaitaires de 5 points de base, pas de
  liquidité ni de frais d'emprunt.
- **Contamination des LLM en backtest**, décrite plus haut.
- **Rapports non standard** : JPMorgan n'utilise pas les titres « Item » habituels ; son rapport de
  gestion n'est pas isolé et seuls les 80 000 premiers caractères sont lus.
- Données de dark pools et de flux d'options collectées mais pas évaluées dans le modèle.

## Suite prévue

1. Exécuter le bras B de l'expérience avec les modèles cibles.
2. Laisser tourner le paper-trading plusieurs mois et publier la courbe, quelle qu'elle soit.
3. Lire l'ensemble des dépôts de l'univers avec les modèles cibles, puis relancer `run_filings_study.py` :
   c'est à cette échelle que l'utilité prédictive des variables pourra être jugée.

---

*Projet personnel à but éducatif. Rien ici ne constitue un conseil en investissement.*
