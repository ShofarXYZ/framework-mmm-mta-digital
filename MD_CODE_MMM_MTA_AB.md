# PROMPT PARA CLAUDE CODE
## App Streamlit — Simulador MMM x MTA x Testes A/B (Digital Analytics)

Copie e cole o bloco abaixo inteiro no Claude Code (terminal, VS Code ou app desktop), na raiz de uma pasta vazia do projeto.

---

```
Você vai construir um aplicativo Streamlit multi-página em Python que materializa,
com dados reais e Machine Learning, o framework "MMM x MTA — Framework de Digital
Analytics" (documento de referência: Framework_MMM_x_MTA_Digital_Analytics.pptx,
já está na pasta do projeto em ./reference/Framework_MMM_x_MTA_Digital_Analytics.pptx
— leia esse arquivo primeiro para entender a linguagem, os conceitos e a ordem do
framework antes de codar).

O objetivo final: um analista de Digital Analytics abre o app no dia a dia e navega
por páginas que reproduzem, na prática, cada camada do framework: Marketing Mix
Modeling (MMM), Multi-Touch Attribution (MTA/atribuição), e Testes A/B alimentados
pelos dois anteriores — fechando o loop de governança MMM ⇄ MTA ⇄ A/B ⇄ Learning
Repository que está no PPT.

=====================================================================
1. DADOS DISPONÍVEIS (já estão em ./data/, use exatamente esses arquivos)
=====================================================================

a) data/mmm_dataset.csv  (209 linhas, granularidade SEMANAL)
   Colunas: date (DD-MM-AAAA), holiday (0/1), sales_promotion (texto: Normal/BOGO/etc),
   competitor_spend, instagram_spend, google_ads_spend, tv_spend, youtube_spend,
   newspaper_spend, influencer_spend, ott_spend, sales (variável dependente).
   OBS: há valores ausentes em quase todas as colunas de spend (entre 50 e 80 linhas
   faltantes por canal) e 3 linhas faltantes em "sales" — trate isso explicitamente
   (ver seção 4a, Data Quality). Use este dataset como fonte única da página de MMM.
   Contém canais OFFLINE (tv_spend, newspaper_spend) e DIGITAIS (instagram_spend,
   google_ads_spend, youtube_spend, influencer_spend, ott_spend) juntos — isso é
   proposital: a página de MMM deve deixar claro que o modelo mistura os dois mundos.

b) data/digital_marketing_campaign_dataset.csv  (8.000 linhas, granularidade
   CLIENTE/CAMPANHA, sem timestamp de sequência de eventos)
   Colunas relevantes: CustomerID, Age, Gender, Income, CampaignChannel
   (Social Media, Email, PPC, Referral, SEO), CampaignType (Awareness, Retention,
   Conversion, Consideration), AdSpend, ClickThroughRate, ConversionRate,
   WebsiteVisits, PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks,
   PreviousPurchases, LoyaltyPoints, Conversion (0/1, alvo binário).
   Use este dataset como fonte principal da página de MTA: como não há sequência de
   touchpoints por usuário nativa, você deve SIMULAR jornadas multi-touch de forma
   transparente (ver seção 4b) e deixar isso documentado na UI — não finja que é
   dado de clickstream real.

c) data/marketing_campaign_dataset.csv  (200.000 linhas, granularidade CAMPANHA)
   Colunas: Campaign_ID, Company, Campaign_Type, Target_Audience, Duration,
   Channel_Used (Google Ads, YouTube, Instagram, Website, Facebook, Email),
   Conversion_Rate, Acquisition_Cost (string "$X,XXX.00" — converter para float),
   ROI, Location, Language, Clicks, Impressions, Engagement_Score, Customer_Segment,
   Date (2021, diário).
   Use este dataset como fonte da página de Testes A/B (comparação de canal x canal,
   campanha x campanha) e como validação cruzada da página de MTA (ranking de canais
   por ROI/Conversion_Rate real vs. crédito atribuído pelos modelos de MTA).

=====================================================================
2. ARQUITETURA DO APP
=====================================================================

Use a API nativa de multi-página do Streamlit (st.Page + st.navigation), Python 3.11+.
Estrutura de pastas:

  app.py                      -> entrypoint, monta st.navigation, config global, tema
  requirements.txt
  reference/
    Framework_MMM_x_MTA_Digital_Analytics.pptx
  data/
    mmm_dataset.csv
    digital_marketing_campaign_dataset.csv
    marketing_campaign_dataset.csv
  src/
    data_loader.py            -> funções cacheadas (st.cache_data) de leitura/limpeza
    mmm/
      transforms.py           -> adstock, saturação (Hill/S-curve), log-log/log-linear
      model.py                 -> fit (Ridge/ElasticNet + opção Bayesian via PyMC),
                                   contribution decomposition, response curves
      optimizer.py             -> otimização de budget (scipy.optimize.minimize,
                                   restrição de orçamento total, bounds por canal)
    mta/
      heuristics.py             -> first-click, last-click, linear, time-decay,
                                    position-based (U-shaped)
      journey_sim.py            -> simulação de jornadas multi-touch a partir do
                                    dataset de cliente único (ver seção 4b)
      markov.py                  -> Markov Chain + Removal Effect (crédito algorítmico)
      shapley.py                  -> Shapley Value entre canais (coalizões)
    abtest/
      power.py                    -> cálculo de duração/tamanho de amostra
                                      (equivalente à aba "Calculator | AB Test" do PPT)
      frequentist.py               -> z-test de proporções, qui-quadrado, IC
      bayesian.py                   -> Beta-Binomial, probabilidade de ser vencedor
      sequential.py                  -> SPRT simplificado
      geo_holdout.py                  -> simulação de geo-experiment/holdout sobre
                                         o mmm_dataset (diferença-em-diferenças +
                                         comparação com o lift previsto pelo MMM)
    viz/
      charts.py                    -> wrappers Plotly padronizados (tema do app)
    utils/
      styling.py                   -> paleta de cores, CSS custom, componentes
  pages/
    0_🏠_Home.py
    1_📊_MMM_Explorer.py
    2_🧪_MMM_Modelagem.py
    3_💰_MMM_Otimizador_de_Budget.py
    4_🔀_MTA_Modelos_de_Atribuicao.py
    5_🕸️_MTA_Markov_e_Shapley.py
    6_🧬_Teste_AB_Calculadora.py
    7_📈_Teste_AB_Simulador_e_Resultados.py
    8_🌍_Teste_AB_Geo_Holdout_MMM.py
    9_🔁_Loop_de_Governanca_e_Learning_Repository.py

=====================================================================
3. PÁGINAS — REQUISITOS FUNCIONAIS DETALHADOS
=====================================================================

--------------------------------------------------------------------
PÁGINA 0 — Home
--------------------------------------------------------------------
- Resumo executivo do framework (reaproveite a linguagem/estrutura do PPT: MMM =
  estratégico/top-down, MTA = tático/bottom-up, Testes A/B = validação causal).
- Diagrama do loop de governança (MMM ⇄ MTA ⇄ Testes A/B ⇄ Learning Repository)
  como componente visual (pode ser graphviz ou plotly com shapes).
- Cards de KPIs rápidos calculados on-the-fly a partir dos 3 datasets (ex: total
  investido, sales total, conversion rate médio, ROI médio) para dar vida à Home.
- Navegação clara para as demais páginas, com 1 frase explicando o "quando usar"
  de cada uma (reaproveite a matriz de decisão do PPT).

--------------------------------------------------------------------
PÁGINA 1 — MMM Explorer (Exploração e Data Quality)
--------------------------------------------------------------------
Fonte: mmm_dataset.csv
- Parse robusto de "date" (formato DD-MM-AAAA, alguns registros fora de ordem —
  ordene por data).
- Seção de Data Quality: heatmap/tabela de missing values por coluna, com os
  MÉTODOS de tratamento do PPT selecionáveis via radio/selectbox por canal:
  "Imputação (média/mediana)", "Forecast (média móvel 4 semanas)", "Replace com
  zero", "Interpolação linear". Aplique a escolha do usuário e mostre antes/depois.
- Detecção de outliers (z-score ou IQR) com explicação contextual (feriado,
  promoção) cruzando com as colunas holiday/sales_promotion.
- Análise univariada (histograma, boxplot) e bivariada (scatter spend x sales,
  correlação) por canal, com seletor de canal.
- Gráfico de série temporal de "sales" com marcação de feriados/promoções
  sobrepostas (linha + rug plot).

--------------------------------------------------------------------
PÁGINA 2 — MMM Modelagem (o motor estatístico)
--------------------------------------------------------------------
Fonte: dataset limpo da Página 1 (persistir em st.session_state).
- Sidebar de configuração do modelo:
  * Forma funcional: Linear / Log-Linear / Log-Log (implemente as 3 do PPT)
  * Adstock: slider de decay (0 a 1) por canal, ou botão "otimizar automaticamente"
    (grid search minimizando erro)
  * Saturação: toggle Hill/S-curve com parâmetros ajustáveis (half-saturation, slope)
  * Regularização: Ridge / Lasso / ElasticNet (sklearn), com slider de alpha
- Rodar modelo (sklearn Ridge/ElasticNet ou, se PyMC estiver instalado, oferecer
  toggle "Modo Bayesiano" com priors simples e mostrar incerteza dos coeficientes
  via HDI — implemente com try/except import para não quebrar se pymc não estiver
  disponível, e nesse caso mostrar aviso e cair para modo frequentista)
- Outputs obrigatórios:
  * Tabela de coeficientes com sinal, magnitude e interpretação em texto
  * R², Adjusted R², MAPE, MAE em holdout temporal (últimas 8-12 semanas)
  * VIF (multicolinearidade) por variável
  * Gráfico de decomposição (contribution analysis): área empilhada Base vs. cada
    canal de mídia ao longo do tempo, somando 100% do "sales" previsto
  * Curvas de resposta (resposta marginal por canal: eixo X = investimento,
    eixo Y = contribuição incremental), permitindo visualizar saturação
  * Due-to analysis simplificada: comparação YoY/período a período da contribuição
    de cada driver (se houver dado suficiente; senão QoQ dentro do próprio dataset)
- Cada canal digital (instagram, google_ads, youtube, influencer, ott) deve ter uma
  anotação lateral lembrando que ele também é otimizado taticamente via MTA/DDA —
  reforçando o conceito do PPT de que MMM não substitui MTA, complementa.

--------------------------------------------------------------------
PÁGINA 3 — MMM Otimizador de Budget
--------------------------------------------------------------------
- A partir do modelo ajustado na Página 2 (usar de session_state; se não existir,
  orientar o usuário a rodar a Página 2 primeiro).
- Inputs: orçamento total disponível (slider/number_input), bounds mínimo/máximo
  por canal (ex: não pode zerar TV, não pode passar de X em Influencer).
- Rodar scipy.optimize.minimize (SLSQP) maximizando sales previsto sujeito à
  restrição de soma = orçamento total, usando as curvas de resposta/saturação
  ajustadas.
- Output: tabela "alocação atual vs. alocação ótima" + gráfico de barras
  comparativo + "lift esperado (%)" em destaque grande.
- Cenário "E se": slider por canal para simular manualmente (+/- % de investimento)
  e ver o impacto estimado no sales, sem re-otimizar — resposta instantânea.

--------------------------------------------------------------------
PÁGINA 4 — MTA: Modelos de Atribuição (heurísticos + DDA-like)
--------------------------------------------------------------------
Fonte: digital_marketing_campaign_dataset.csv
- IMPORTANTE: no topo da página, um bloco explicativo (st.info) deixando claro que
  o dataset é por cliente/campanha (não é uma sequência real de cliques), então o
  app vai SIMULAR uma jornada multi-touch por cliente concatenando
  CampaignChannel + CampaignType como pseudo-touchpoints, ponderados por
  WebsiteVisits/EmailOpens/EmailClicks/SocialShares como proxy de intensidade de
  interação, terminando em Conversion. Documente a lógica de simulação no código
  (journey_sim.py) com comentários claros.
- Construir, por cliente, uma "jornada sintética" ordenável (ex: Awareness antes de
  Consideration antes de Conversion, ponderada pelas colunas de engajamento) —
  sim, aqui você tem liberdade de engenharia, mas precisa ser DEFENSÁVEL e
  documentado na tela.
- Implementar e comparar lado a lado, com os MESMOS dados:
  first-click, last-click, linear, time-decay, position-based (U-shaped 40/20/40).
- Visualização: gráfico de barras comparando % de crédito por canal entre os 5
  modelos (grouped bar chart) — deixe visualmente óbvio como last-click distorce
  vs. os demais.
- Tabela resumo: canal | crédito por modelo | AdSpend total | CPA implícito por
  modelo (AdSpend / conversões creditadas).

--------------------------------------------------------------------
PÁGINA 5 — MTA: Markov Chain e Shapley Value (crédito algorítmico)
--------------------------------------------------------------------
- Sobre a mesma jornada sintética da Página 4:
  * Markov Chain: construir matriz de transição entre estados (canais + Start/
    Conversion/Null), calcular probabilidade de conversão total, e o Removal
    Effect de cada canal (retirar o canal, recalcular probabilidade, a diferença
    é o crédito causal aproximado). Pode implementar na mão com numpy/pandas
    (não precisa de lib externa) ou usar a lib "channel_attribution"/pychattr se
    quiser — com fallback manual caso a lib não esteja disponível.
  * Shapley Value: implementar cálculo por amostragem de permutações (Monte Carlo)
    já que o número de canais é pequeno (5), permitindo cálculo exato ou quase-
    exato.
- Gráfico de rede/sankey mostrando o fluxo de transição entre canais até a
  conversão (plotly Sankey).
- Comparação final: tabela com TODOS os modelos de atribuição (heurísticos da
  Página 4 + Markov + Shapley) lado a lado — essa tabela é o "DDA caseiro" do app.

--------------------------------------------------------------------
PÁGINA 6 — Teste A/B: Calculadora (Duração e Amostra)
--------------------------------------------------------------------
Recrie em Python/Streamlit as 3 calculadoras da aba "Calculator | AB Test" do
material de referência do usuário:
  a) Duration Estimation: inputs = nº de variações, usuários/dia, % de tráfego
     alocado ao teste, taxa de conversão atual, uplift esperado (%) → output =
     dias estimados de duração.
  b) Mid-range Impact Estimation: inputs = total de sessões, % alocado ao
     experimento, conversão total, uplift → output = conversões incrementais
     esperadas, taxa final projetada.
  c) Sample Size / Duration V2: inputs = conversão atual, uplift esperado, nº de
     variações, visitantes médios/dia, nível de confiança (80%/90%/95%) →
     output = número de visitantes necessários e dias.
- IMPORTANTE: pré-popular os inputs de "conversão atual" e "uplift esperado" com
  valores REAIS puxados do digital_marketing_campaign_dataset.csv (ex: média de
  ConversionRate do canal selecionado) — conectando a calculadora ao dado real,
  não deixando tudo manual como na planilha original.
- Fórmulas: use a fórmula padrão de tamanho de amostra para teste de duas
  proporções (z_alpha/2 + z_beta, variância de p1/p2) — documente a fórmula em um
  expander "Ver a matemática por trás".

--------------------------------------------------------------------
PÁGINA 7 — Teste A/B: Simulador e Resultados (Significância)
--------------------------------------------------------------------
Fonte: digital_marketing_campaign_dataset.csv e/ou marketing_campaign_dataset.csv
- Modo 1 "Dados reais": usuário escolhe duas categorias para comparar (ex:
  CampaignChannel = "Social Media" vs "PPC", ou Channel_Used = "Google Ads" vs
  "Facebook") e o app calcula automaticamente: visitantes, conversões, taxa de
  conversão, e roda:
  * Teste Z de proporções (frequentista) → z-score, p-value, IC 90/95/99%
  * Teste Bayesiano (Beta-Binomial) → probabilidade de B > A, gráfico de
    distribuições posteriores sobrepostas
  * Teste Sequencial (SPRT simplificado) → mostrar se já daria para decidir antes
    do fim do período com base nos dados acumulados
  Reproduza a mesma lógica/estética da "A/B Testing Significance Calculator" do
  material de referência (Control vs Variation, IC 90/95/99%, Z-score, P-value,
  "Significant At: 90%/95%/99% → YES/NO").
- Modo 2 "Simulação manual": inputs manuais de visitantes/conversões por variante
  (como a calculadora original), para quando o analista quiser simular um cenário
  hipotético antes do teste rodar.
- Classificar o resultado como Winner / Neutral / Loser automaticamente (regra:
  significativo E lift positivo = Winner; significativo E lift negativo = Loser;
  não significativo = Neutral) — reaproveitando a nomenclatura da aba
  "Analytics | Results" do framework do usuário.

--------------------------------------------------------------------
PÁGINA 8 — Teste A/B: Geo-Holdout ligado ao MMM
--------------------------------------------------------------------
Esta é a página que fecha o ciclo MMM → Teste A/B do PPT.
Fonte: mmm_dataset.csv + o modelo ajustado na Página 2 (session_state)
- Como não há dado geográfico real no mmm_dataset, simule um "holdout temporal":
  o usuário escolhe um canal (ex: google_ads_spend) e uma janela de semanas
  consecutivas para simular como "período de holdout" (redução simulada do
  investimento em X%).
- Compare: (a) o "sales" real observado nessas semanas, (b) o "sales" contrafactual
  previsto pelo modelo MMM da Página 2 SEM a redução (cenário base), e (c) o
  "sales" com a redução aplicada — apresente a diferença como o "lift incremental
  medido" do canal, e compare com o coeficiente/curva de resposta que o MMM havia
  estimado para aquele canal.
- Se a diferença entre o lift medido no holdout e o lift previsto pelo MMM for
  grande, exibir um alerta: "Recalibrar o modelo — o MMM pode estar super ou
  subestimando este canal", reforçando o loop de aprendizado do framework.
- Implementar isso com uma versão simples de diferença-em-diferenças (DiD).

--------------------------------------------------------------------
PÁGINA 9 — Loop de Governança e Learning Repository
--------------------------------------------------------------------
- Um "repositório de aprendizados" navegável e PERSISTENTE dentro da sessão
  (st.session_state, com opção de exportar/baixar como CSV) onde cada página
  anterior pode "enviar" um registro (botão "📌 Salvar este resultado no
  Learning Repository" em cada página das seções MMM/MTA/A-B).
- Estrutura do registro (espelhando as abas do framework original do usuário):
  experiment_id, origem (MMM/MTA/Geo-Holdout/Teste A/B), canal/driver, hipótese,
  resultado (lift %, p-value, winner/neutral/loser), insight, próximo passo
  sugerido, data.
- Tabela final com filtro por origem/canal/resultado, e um gráfico de "funil do
  framework" mostrando quantos itens estão em cada etapa (Opportunity → Hypothesis
  → Priorization → Design → Tracking → Results → Learning) — pode ser um funnel
  chart do Plotly.
- Botão de export para CSV no MESMO formato de colunas da planilha original do
  usuário (Roadmap Teste A/B), para que o analista possa colar de volta na
  planilha real se quiser.

=====================================================================
4. DETALHES TÉCNICOS OBRIGATÓRIOS
=====================================================================

--- 4a. Data Quality (MMM) ---
- Ler mmm_dataset.csv com parse de data DD-MM-AAAA (dayfirst=True), ordenar.
- Tratar sales_promotion como categórica (criar dummies: is_bogo, is_normal, etc.)
- Reportar % de missing por coluna ANTES de qualquer imputação, sempre visível.
- Nunca sobrescrever o CSV original — todo tratamento é em memória/session_state.

--- 4b. Simulação de jornada MTA (documentar isso claramente na página 4) ---
Sugestão de heurística (ajustável, mas precisa ser algo así):
  1. Para cada CustomerID, gerar de 1 a 4 touchpoints sintéticos:
     - Sempre inclui o CampaignChannel/CampaignType real da linha como o
       touchpoint "principal" (mais próximo da conversão).
     - Se PreviousPurchases > 0 ou LoyaltyPoints alto, adicionar um touchpoint
       anterior sintético de canal diferente (ex: Email ou SEO) representando
       jornada de retenção.
     - Se EmailOpens/EmailClicks > 0 mas CampaignChannel != Email, adicionar
       Email como touchpoint intermediário.
     - Ordenar por proximidade à conversão de forma determinística e documentada.
  2. A conversão final = coluna Conversion (0/1).
  Deixe a lógica isolada em journey_sim.py com uma docstring completa explicando
  a decisão de design, e um expander na UI "Como a jornada foi simulada?".

--- 4c. Adstock e Saturação (MMM) ---
- Adstock geométrico: x_t_adstocked = x_t + decay * x_{t-1}_adstocked
- Saturação Hill: y = x^s / (x^s + k^s), parâmetros s (slope) e k (half-saturation)
  ajustáveis via slider, com preview do formato da curva antes de aplicar ao
  modelo.

--- 4d. Estatística de Teste A/B ---
- Duas proporções (z-test): usar statsmodels.stats.proportion.proportions_ztest
  e proportion_confint. Não reimplementar na mão se statsmodels estiver
  disponível; se não estiver, incluir fallback manual com scipy.stats.norm.
- Bayesiano: Beta(1,1) como prior não-informativo, posterior Beta(alpha+conv,
  beta+não-conv), amostragem via numpy para P(B>A).
- Tamanho de amostra: fórmula clássica com z_alpha e z_beta (poder estatístico),
  parametrizável por nível de confiança e poder (80%/90%).

=====================================================================
5. STACK E DEPENDÊNCIAS (requirements.txt)
=====================================================================
streamlit>=1.38
pandas
numpy
scikit-learn
scipy
statsmodels
plotly
python-pptx        # para ler texto do PPT de referência na Home, se aplicável
pymc               # opcional — envolva em try/except em todo import
arviz              # opcional, junto com pymc

=====================================================================
6. UX / DESIGN
=====================================================================
- Tema visual consistente com o PPT de referência: navy (#21295C), teal (#1C7293),
  azul (#065A82), dourado (#E8A33D) como accent. Configure via .streamlit/config.toml
  (theme.primaryColor, backgroundColor etc.) e reforce com CSS custom em
  utils/styling.py.
- Cada página deve abrir com: título, 1 frase de contexto ("onde isso está no
  framework"), e um badge/caption indicando a camada (Estratégico=MMM /
  Tático=MTA / Validação=Teste A/B), igual à lógica de camadas do PPT.
- Use st.tabs() dentro de páginas densas em vez de acumular tudo em scroll único.
- Todo gráfico em Plotly (não matplotlib puro), interativo, com hover informativo.
- Sidebar global (via app.py) com um seletor de "Projeto/Segmento" fake (ex: filtro
  por Customer_Segment ou Target_Audience) para dar sensação de ferramenta real.

=====================================================================
7. QUALIDADE E ENTREGA
=====================================================================
- Toda função pesada (leitura de CSV, fit de modelo) deve usar st.cache_data ou
  st.cache_resource apropriadamente.
- Tratar erros de dado ausente/formato com try/except e mensagens amigáveis
  (st.warning/st.error), nunca deixar o app quebrar com traceback cru.
- Adicionar um README.md explicando: como rodar (`streamlit run app.py`), a
  origem dos 3 datasets, o mapeamento página → conceito do framework, e as
  limitações assumidas (principalmente a simulação de jornada da Página 4/5 e
  do holdout geográfico da Página 8 — deixe claro que são aproximações
  pedagógicas para simular o framework, não tracking real).
- Ao final, rode o app localmente (`streamlit run app.py --server.headless true`)
  e valide que todas as 10 páginas carregam sem erro antes de finalizar.

Comece pela leitura do PPT de referência e dos 3 CSVs para confirmar o
entendimento, proponha brevemente a arquitetura de pastas, e então implemente
incrementalmente: (1) data_loader + Home, (2) MMM Explorer + Modelagem +
Otimizador, (3) MTA Atribuição + Markov/Shapley, (4) Teste A/B (3 páginas),
(5) Loop de Governança. Rode e teste cada bloco antes de seguir para o próximo.
```

---

## Como usar este prompt

1. Crie uma pasta de projeto vazia (ex: `mmm-mta-streamlit/`).
2. Dentro dela, crie a subpasta `reference/` e copie o
   `Framework_MMM_x_MTA_Digital_Analytics.pptx` para lá.
3. Crie a subpasta `data/` e copie os 3 CSVs (`mmm_dataset.csv`,
   `digital_marketing_campaign_dataset.csv`, `marketing_campaign_dataset.csv`)
   para lá.
4. Abra o Claude Code nessa pasta e cole o bloco de comando acima.
5. Acompanhe a execução por etapas (o prompt já pede que o Claude Code construa
   incrementalmente e teste cada bloco antes de seguir).

## Mapa rápido: dataset → página → conceito do framework

| Dataset | Página(s) que usa | Conceito do PPT |
|---|---|---|
| `mmm_dataset.csv` (semanal, TV/jornal/Instagram/Google/YouTube/Influencer/OTT + sales) | MMM Explorer, MMM Modelagem, MMM Otimizador, Geo-Holdout | Camada estratégica — adstock, saturação, regressão, curvas de resposta, otimização de budget |
| `digital_marketing_campaign_dataset.csv` (cliente/campanha, 5 canais digitais) | MTA Atribuição, MTA Markov/Shapley | Camada tática — first/last-click, linear, time-decay, position-based, Markov Chain, Shapley (equivalente ao DDA) |
| `marketing_campaign_dataset.csv` (200k campanhas, ROI/Engagement por canal) | Teste A/B Simulador, validação cruzada da MTA | Camada de validação — comparação canal x canal, ROI real vs. crédito atribuído |

## Observação sobre os dados

Nenhum dos 3 CSVs contém uma sequência real de touchpoints por usuário (clickstream
com timestamp) nem dados de geolocalização — por isso o prompt instrui explicitamente
o Claude Code a **simular e documentar** a jornada multi-touch (página MTA) e o
holdout geográfico (página 8), de forma transparente na UI. Isso é intencional: o
objetivo do app é pedagógico/demonstrativo do framework, reproduzindo a lógica de
MMM x MTA x Teste A/B com dados públicos — quando você plugar o BigQuery real
(conforme a arquitetura técnica do PPT, `fct_mmm_weekly` e `fct_mta_touchpoint`), a
mesma estrutura de páginas passa a rodar sobre jornadas reais, sem necessidade de
simulação.
