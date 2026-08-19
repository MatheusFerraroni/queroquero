# Datasets

Inventário agregado dos corpora. Relatórios não contêm textos, usernames, IDs ou
URLs dos registros.

| Dataset | Fonte selecionada | Inventário confirmado | Principal pendência |
| --- | --- | ---: | --- |
| [Adrenaline](adrenaline.md) | `conversations.zip` | 4.896.419 entradas; 76,1 GB | validar schema |
| [OuterSpace](outerspace.md) | `conversations.zip` | 3.650.595 entradas; 5,8 GB | validar schema |
| [BrWaC-CLEAN](brwac.md) | `BrWac.zip` → `data/*.txt` | 3.063.728 textos | medir tokens/qualidade |
| [MultiWOZ-PTBR](multiwoz-ptbr.md) | 17 JSONs | 8.437 diálogos | definir projeção/dedup |
| [WackyWacky](wackywacky.md) | `pages.tsv` | 56,2 GB; linhas desconhecidas | passagem sequencial |
| [GigaVerbo-v2](gigaverbo.md) | `default/train` remoto | 372.108.576 linhas | decisão de ativação |

`forum.*.zip`, `messages.zip` e `names.tsv` não são texto de treino.
`conversations_min.zip` serve apenas como fixture de parser.

## Regras comuns

- fontes são somente leitura e identificadas por fingerprint;
- cada dataset tem budget, manifesto, métricas e proveniência próprios;
- preparação gera shards locais pequenos e retomáveis;
- reexecuções reutilizam derivados validados;
- conteúdo e metadados sensíveis não entram no Git ou em logs.

Ainda faltam budgets finais, filtros de qualidade, tokenização e políticas de
deduplicação. Nenhum valor deve ser inferido silenciosamente.
