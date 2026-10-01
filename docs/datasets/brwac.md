# BrWaC-CLEAN

Inventário de 2026-08-18, feito somente com metadados e amostras limitadas.

| Item | Valor |
| --- | ---: |
| `BrWac.zip` | 5.837.490.062 bytes |
| entradas totais | 3.063.730 |
| `data/*.txt` | 3.063.728 documentos |
| bytes descompactados | 12.574.567.129 |
| `names.tsv` descompactado | 482.810.058 bytes |

A fonte de treino é exclusivamente `data/*.txt`. `names.tsv` é metadado e não
é lido como texto.

Uma amostra histórica de 11 documentos abriu como UTF-8, sem estrutura
JSON/XML reconhecível. Essa observação não mede qualidade, duplicação ou
outliers do corpus completo.

## Adapter

O adapter manifesta o diretório central e lê documentos selecionados
diretamente do ZIP, sempre como UTF-8 estrito. O ZIP completo nunca é extraído.
A seleção usa o hash estável do caminho do membro e budgets independentes por
perfil, evitando depender da ordem física do arquivo.

O fingerprint agrega nome, CRC e tamanhos de todos os membros, além do tamanho
do ZIP. O cursor registra apenas a posição na seleção e permite retomada quando
fonte e configuração continuam idênticas. A limpeza, deduplicação exata,
tokenização, split e packing são aplicados pelo núcleo comum.

Como a fonte contém pontuação previamente separada por espaços, o filtro
versionado `detokenize_brwac_v1` remove somente espaços antes de vírgula/ponto e
dentro de parênteses. Hífens e os demais símbolos permanecem inalterados.

```sh
python -m queroquero.prepare run --dataset brwac --profile smoke
```

As métricas e a proveniência usam hashes e contagens; conteúdo dos documentos e
caminhos absolutos não são persistidos nos shards.
