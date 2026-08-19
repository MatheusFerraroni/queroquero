# BrWaC-CLEAN

Inspeção de 2026-08-18, somente de metadados e amostras limitadas.

| Item | Valor |
| --- | ---: |
| `BrWac.zip` | 5.837.490.062 bytes |
| entradas totais | 3.063.730 |
| `data/*.txt` | 3.063.728 documentos |
| bytes descompactados | 12.574.567.129 |
| `names.tsv` descompactado | 482.810.058 bytes |

A fonte selecionada é somente `data/*.txt`. `names.tsv` é metadado e não entra
como texto de treino.

Uma amostra de 11 documentos abriu como UTF-8, sem estrutura JSON ou tags
HTML/XML reconhecíveis. Isso não mede qualidade, duplicação ou outliers do
corpus completo.

## Antes da preparação

1. manifestar os membros e fingerprints do ZIP;
2. amostrar tamanhos diferentes de forma determinística;
3. medir tokens com o tokenizer do modelo;
4. definir limites, limpeza, deduplicação e budget;
5. gerar shards retomáveis preservando `archive` e `member_path`.

Não extrair o ZIP inteiro.
