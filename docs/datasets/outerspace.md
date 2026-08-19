# OuterSpace

Inspeção de 2026-08-18, somente de metadados e amostras limitadas.

| Arquivo | Tamanho | Entradas | Uso |
| --- | ---: | ---: | --- |
| `forum.outerspace.com.br.zip` | 4.800.738.537 bytes | 570.652 | não selecionado |
| `conversations.zip` | 5.778.504.549 bytes | 3.650.595 | selecionado |
| `conversations_min.zip` | — | — | fixture de parser |

O ZIP de fórum contém categorias, tópicos e mensagens em JSON, com um outlier
próximo de 495 MB descompactado. No ZIP selecionado foram observados 1.406.509
TSVs sob `clear_threads/` antes da interrupção; schema, encoding e total
descompactado continuam desconhecidos.

## Antes da preparação

1. gerar o manifesto do ZIP no host dos dados;
2. validar poucas entradas por faixa de tamanho;
3. confirmar unidade documental, colunas, encoding e remoção de HTML/PII;
4. definir budget, filtros e deduplicação;
5. processar por shard com cursor e proveniência.

Não extrair os ZIPs completos nem inferir o schema da enumeração parcial.
