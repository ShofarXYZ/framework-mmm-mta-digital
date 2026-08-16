"""Data-Driven Attribution no espírito do modelo do GA4.

O QUE O GA4 FAZ (e o que o app reproduz)
----------------------------------------
Desde 2023 o DDA é o modelo padrão do GA4. O algoritmo do Google é fechado, mas
a documentação descreve com clareza os três pilares que o diferenciam dos
modelos de regra fixa — e são esses três que esta implementação reproduz:

1. **Usa jornadas que converteram E jornadas que não converteram.** É o pilar
   central. Todos os outros modelos do app (first-click, last-click, linear,
   time-decay, position-based, Markov, Shapley) só olham quem converteu. Sem o
   grupo de comparação não há como saber se um canal aparece nas conversões
   porque ele causa conversão ou porque ele simplesmente aparece em tudo.

2. **Aprende os pesos com o próprio dado**, em vez de aplicar percentuais fixos.
   Aqui isso é uma regressão logística que prevê a probabilidade de conversão a
   partir da composição da jornada.

3. **Considera a posição e a composição** do touchpoint, não só a presença:
   ser o primeiro contato, ser o último, e que fatia da jornada aquele canal
   ocupa entram como variáveis separadas.

COMO O CRÉDITO É REPARTIDO (contrafactual por jornada)
-----------------------------------------------------
Para cada conversão real, pergunta-se canal a canal:
*"qual era a probabilidade de esta pessoa converter COM este canal na jornada, e
qual seria SEM ele?"* A queda de probabilidade é a contribuição marginal daquele
canal naquela jornada específica. As contribuições são normalizadas para somar
1 (uma conversão = um crédito) e somadas por canal.

É a mesma pergunta do Removal Effect do Markov, mas feita **por jornada** e com
a probabilidade vindo de um modelo que enxergou também quem não converteu.

LIMITE HONESTO
--------------
Isto não é o algoritmo do Google — é uma reconstrução dos princípios que ele
publica, sobre a jornada sintética deste app. Serve para entender o raciocínio e
comparar com as réguas de regra fixa. E, como todo modelo de atribuição, mede
correlação: a prova de causa continua sendo o experimento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _feature_matrix(paths: list[tuple[str, ...]], channels: list[str]) -> np.ndarray:
    """Composição da jornada em números, por linha.

    Para cada canal: presença, é o primeiro, é o último e que fatia da jornada
    ele ocupa. Mais o tamanho da jornada. É o que permite ao modelo aprender que
    um canal vale diferente conforme a posição — sem pesos fixos.
    """
    n = len(paths)
    n_ch = len(channels)
    index = {c: i for i, c in enumerate(channels)}
    features = np.zeros((n, n_ch * 4 + 1), dtype=float)

    for row, path in enumerate(paths):
        length = len(path)
        features[row, -1] = length
        if length == 0:
            continue
        for position, channel in enumerate(path):
            i = index.get(channel)
            if i is None:
                continue
            features[row, i] = 1.0                                   # presença
            features[row, n_ch + i] += 1.0 / length                  # fatia da jornada
            if position == 0:
                features[row, 2 * n_ch + i] = 1.0                    # primeiro contato
            if position == length - 1:
                features[row, 3 * n_ch + i] = 1.0                    # último contato
    return features


def _paths_without(paths: list[tuple[str, ...]], channel: str) -> list[tuple[str, ...]]:
    return [tuple(c for c in path if c != channel) for path in paths]


def dda_attribution(journeys: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """Crédito por canal no modelo Data-Driven (estilo GA4).

    Returns:
        DataFrame indexado por canal com:
        conversoes_creditadas | share_% | contribuicao_media | presenca_em_conversoes_%
    """
    if channels is None:
        channels = sorted({c for path in journeys["path"] for c in path})

    paths = [tuple(p) for p in journeys["path"]]
    y = journeys["converted"].astype(int).to_numpy()

    # 1. Modelo treinado com TODAS as jornadas — convertidas e não convertidas.
    X = _feature_matrix(paths, channels)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)
    model.fit(scaler.transform(X), y)

    # 2. Contrafactual: probabilidade com e sem cada canal, jornada a jornada.
    converted_mask = y == 1
    conv_paths = [p for p, keep in zip(paths, converted_mask) if keep]
    if not conv_paths:
        return pd.DataFrame(columns=["conversoes_creditadas", "share_%"])

    X_conv = _feature_matrix(conv_paths, channels)
    p_full = model.predict_proba(scaler.transform(X_conv))[:, 1]

    n_ch = len(channels)
    deltas = np.zeros((len(conv_paths), n_ch), dtype=float)
    for i, channel in enumerate(channels):
        # Contrafactual ISOLADO: apaga só as variáveis do canal i (presença,
        # fatia, primeiro, último) e mantém o resto da jornada intacto.
        #
        # A alternativa óbvia — reconstruir a jornada sem o canal — parece mais
        # natural, mas contamina a medida: encurtar o caminho muda o tamanho da
        # jornada E redistribui a fatia dos outros canais, então um canal que
        # aparece em quase toda jornada acaba levando crédito por um efeito de
        # comprimento, não por mérito próprio.
        X_without = X_conv.copy()
        for block in range(4):
            X_without[:, block * n_ch + i] = 0.0
        p_without = model.predict_proba(scaler.transform(X_without))[:, 1]
        deltas[:, i] = p_full - p_without

    presence = np.array([[1.0 if c in path else 0.0 for c in channels] for path in conv_paths])

    # 3. Uma conversão = um crédito, repartido entre os canais presentes conforme
    #    a queda de probabilidade.
    #
    #    O peso passa por um softplus em vez de um corte em zero. Cortar parecia
    #    natural ("contribuição negativa não existe"), mas zera de vez qualquer
    #    canal cujo coeficiente fique levemente negativo — e, quando o dado tem
    #    pouco sinal, isso apaga canais inteiros da régua por causa de ruído.
    #    O softplus é suave: quem contribui mais leva mais, quem contribui menos
    #    leva pouco, e ninguém é eliminado por uma diferença na terceira casa.
    scale = max(float(np.std(deltas[presence > 0])), 1e-4)
    smooth = np.log1p(np.exp(np.clip(deltas / scale, -30, 30))) * presence
    totals = smooth.sum(axis=1)
    flat = presence / np.maximum(presence.sum(axis=1, keepdims=True), 1)
    weights = np.where(totals[:, None] > 1e-12, smooth / np.maximum(totals[:, None], 1e-12), flat)

    credited = weights.sum(axis=0)
    out = pd.DataFrame(
        {
            "canal": channels,
            "conversoes_creditadas": credited,
            "contribuicao_media": np.divide(
                (deltas * presence).sum(axis=0), np.maximum(presence.sum(axis=0), 1)
            ),
            "presenca_em_conversoes_%": 100 * presence.mean(axis=0),
        }
    ).set_index("canal")

    total = out["conversoes_creditadas"].sum()
    out["share_%"] = 100 * out["conversoes_creditadas"] / total if total > 0 else np.nan
    # Diagnóstico de sinal: quanto o modelo consegue separar quem converteu de
    # quem não converteu. Perto de 0,5 significa que a jornada quase não explica
    # a conversão — e aí o DDA se aproxima do Linear, por honestidade, não por erro.
    try:
        from sklearn.metrics import roc_auc_score

        out.attrs["auc"] = float(roc_auc_score(y, model.predict_proba(scaler.transform(X))[:, 1]))
    except Exception:
        out.attrs["auc"] = float("nan")
    out.attrs["n_jornadas"] = int(len(paths))
    out.attrs["n_nao_convertidas"] = int((~converted_mask).sum())
    return out.sort_values("share_%", ascending=False)
