# Exploração dos datasets montados

Esta pasta registra a exploração estrutural dos corpora montados, sem copiar
mensagens, usernames, URLs pessoais ou outros valores dos registros para o
repositório.

Relatórios disponíveis:

- [Adrenaline](adrenaline.md): `forum.adrenaline.com.br.zip` e
  `conversations.zip`;
- [OuterSpace](outerspace.md): `forum.outerspace.com.br.zip` e
  `conversations.zip`;
- [BrWaC-CLEAN](brwac.md): `BrWac.zip`;
- [MultiWOZ-PTBR](multiwoz-ptbr.md): 17 arquivos `dialogues_*.json`;
- [WackyWacky PT-BR](wackywacky.md): `pages.tsv`;
- [GigaVerbo-v2](gigaverbo.md): exploração oficial e condições de
  pré-ativação, sem acesso a registros individuais.

## Inventário confirmado

| Fonte | Artefato | Tamanho | Entradas ou unidades confirmadas |
| --- | --- | ---: | ---: |
| Adrenaline | `forum.adrenaline.com.br.zip` | 2.157.145.545 bytes | 356.799 |
| Adrenaline | `conversations.zip` | 76.123.303.722 bytes | 4.896.419 |
| OuterSpace | `forum.outerspace.com.br.zip` | 4.800.738.537 bytes | 570.652 |
| OuterSpace | `conversations.zip` | 5.778.504.549 bytes | 3.650.595 |
| BrWaC-CLEAN | `BrWac.zip` | 5.837.490.062 bytes | 3.063.730 |
| MultiWOZ-PTBR | 17 arquivos `dialogues_*.json` | 174.167.553 bytes | 8.437 diálogos |
| WackyWacky PT-BR | `pages.tsv` | 56.205.366.282 bytes | 19 colunas; total de linhas não contado |

As contagens dos ZIPs foram validadas pelo cabeçalho com `zipinfo -h`; a do
MultiWOZ veio da leitura estrutural dos 17 JSONs. Elas não significam que todas
as entradas sejam documentos úteis para treino.

## Decisões de fontes

Decisões locais registradas em 2026-08-18; pré-ativação do GigaVerbo-v2
documentada em 2026-08-19:

| Dataset | Fonte selecionada | Uso definido |
| --- | --- | --- |
| BrWaC-CLEAN | `BrWac.zip` → `data/*.txt` | Documentos de texto; `names.tsv` não entra como texto de treino |
| WackyWacky PT-BR | `pages.tsv` | Linhas filtradas com `status == "done"`, `text` e `text_md5` presentes |
| MultiWOZ-PTBR | `dialogues_001.json` a `dialogues_017.json` | Todos os 17 arquivos, preservando diálogos e ordem dos turns |
| OuterSpace | `conversations.zip` | Corpus conversacional completo selecionado para este dataset |
| Adrenaline | `conversations.zip` | Corpus conversacional completo selecionado para este dataset |
| GigaVerbo-v2 | candidato `Polygl0t/gigaverbo-v2`, `default`/`train`, em revisão pinada | Pré-ativação documentada; continua desabilitado até decisão explícita |

Para Adrenaline e OuterSpace, `conversations_min.zip` serve somente como fixture
pequena para desenvolver e testar o parser. Os exports `forum.*.zip` não foram
selecionados como fonte de treino. `messages.zip`, existente apenas no
Adrenaline, também não foi selecionado.

Adrenaline e OuterSpace são datasets completamente distintos. Cada um deve ter
proveniência, manifesto, configuração, métricas e orçamento de documentos ou
tokens próprios. Um não substitui nem representa uma etapa do outro.

Quando existir uma configuração executável de pretraining, ela deve reproduzir
estas escolhas. Uma alteração futura de fonte deve ser explícita e registrada
aqui e na configuração do experimento.

## Conclusão operacional atual

Os arquivos-fonte não devem ser enumerados ou lidos integralmente em toda
execução. O fluxo recomendado é dividido em duas fases:

1. **Preparação única por versão do arquivo**
   - calcular um fingerprint dos arquivos-fonte;
   - gerar um manifesto perto do armazenamento remoto;
   - inspecionar uma amostra determinística e limitada;
   - limpar e normalizar somente o subconjunto selecionado;
   - produzir shards derivados com proveniência e configuração registradas.
2. **Treino e reexecuções**
   - validar o fingerprint e a configuração;
   - reutilizar os manifests e shards derivados;
   - retomar pelo cursor do último shard concluído;
   - nunca depender de nova enumeração de milhões de membros via SSHFS.

O manifesto detalhado, textos processados, shards, caches e índices devem ficar
fora do Git. Apenas schemas, estatísticas agregadas, decisões experimentais e
comandos reproduzíveis devem ser versionados aqui.

## Estado da decisão

Os arquivos-fonte locais estão escolhidos; o GigaVerbo-v2 permanece uma fonte
remota candidata e desabilitada. Ainda não foram definidos a quantidade de
documentos ou tokens, os limites por fonte e todos os filtros de qualidade.

Para o GigaVerbo-v2, a fonte candidata e o filtro mínimo já estão documentados,
mas a ativação ainda depende de orçamento, seed, parâmetros de seleção, revisão
de licenças e configuração executável explícita.

Antes de fixar esses valores, falta validar o parser dos TSVs completos de
Adrenaline e OuterSpace, medir o BrWaC com o tokenizer escolhido, definir a
projeção textual e a deduplicação do MultiWOZ, fazer uma passagem retomável para
contar e filtrar o WackyWacky, estimar texto útil e custos de sanitização, e
preservar a proveniência entre cada fonte e seus derivados.
