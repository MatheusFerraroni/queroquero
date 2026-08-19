# WackyWacky PT-BR: exploração inicial

Data da inspeção: 2026-08-18. O arquivo está em um mount SSHFS somente leitura.
Esta nota registra somente schema, metadados e estatísticas agregadas. Nenhuma
URL, título ou página foi copiada para o repositório.

## Inventário atual

| Arquivo | Formato | Tamanho no mount |
| --- | --- | ---: |
| `wacky/pages.tsv` | TSV descompactado, UTF-8 | 56.205.366.282 bytes |

O arquivo observado anteriormente como `pages.tsv.gz` não está mais presente no
mount. A versão atual é `pages.tsv`, com aproximadamente 52,3 GiB e mtime de
2026-08-18. Essa mudança deve ser considerada ao criar o fingerprint da fonte.

O total de linhas não foi calculado porque isso exigiria uma leitura completa de
56 GB. O arquivo descompactado permite `seek` por byte e retomada eficiente,
diferentemente de um único stream gzip.

## Schema confirmado

A primeira linha é um cabeçalho ASCII seguro com 19 campos:

```text
id
domain_id
parent_page_id
same_as
url
url_md5
url_final
url_final_md5
status_code
title
recursion_level
status
retry_count
text
html
text_md5
html_md5
created_at
updated_at
```

Os nomes indicam uma tabela de estado de um crawler, não apenas um arquivo com
texto pronto para treino. A semântica exata deve ser tratada como provisória até
ser confirmada pela documentação de origem. URLs e títulos são metadados de
proveniência e não devem ser incorporados automaticamente ao texto treinável.

## Amostragem bounded

Foram abertas sete janelas alinhadas por newline nos offsets aproximados de 0%,
10%, 25%, 50%, 75%, 90% e 99% do arquivo. Cada janela foi limitada a 300 linhas,
4 MiB lidos e 1 MiB por linha.

Resultado agregado das 2.100 linhas de dados:

| Medida | Resultado |
| --- | ---: |
| bytes lidos nas linhas amostradas | 5.211.665 |
| quantidade de campos | 19 em todas as linhas |
| erros de UTF-8 | 0 |
| linhas parciais ou acima do limite | 0 |
| tamanho mínimo da linha | 170 bytes |
| mediana | 265 bytes |
| P90 | 6.848 bytes |
| P99 | 37.229 bytes |
| máximo | 45.506 bytes |
| linhas com `text` presente | 621 |
| hashes distintos entre os 621 textos | 621 |
| linhas com `html` presente | 0 |

Não apareceram sequências literais `\\n`, `\\t`, `\\r` nem o caractere de
substituição Unicode nos 621 textos. Na amostra, portanto, cada registro ocupa
uma linha física e o texto já parece normalizado para esse formato.

### Status e disponibilidade de texto

| `status` | Linhas | Com texto | Com `same_as` |
| --- | ---: | ---: | ---: |
| `done` | 808 | 568 | 240 |
| `blocked_language` | 53 | 53 | 0 |
| `blocked_domain` | 113 | 0 | 0 |
| `blocked_limit_recursion` | 900 | 0 | 0 |
| `failed` | 10 | 0 | 0 |
| `failed_timeout` | 12 | 0 | 0 |
| `todo` | 204 | 0 | 0 |

Nos registros `done` amostrados, os 240 sem texto tinham `same_as`, compatível
com referências a outra página em vez de conteúdo duplicado armazenado. Já os
registros `blocked_language` tinham texto, mas não devem ser incluídos apenas
por isso: o próprio status sugere que foram rejeitados pelo filtro de idioma.

A distribuição varia muito pela posição do arquivo. As janelas em 75%, 90% e
99% continham somente `blocked_limit_recursion` e nenhum texto. Portanto, estes
números descrevem a amostra estrutural e não estimam a proporção global de texto
útil. O arquivo aparenta estar ordenado por estado, etapa do crawl ou outra
característica correlacionada.

## Implicações para ingestão

O filtro inicial mais conservador é:

```text
status == "done"
text presente
text_md5 presente
```

Registros sem texto, referências `same_as`, páginas bloqueadas, falhas e itens
pendentes não devem entrar no corpus textual por padrão. `text_md5` pode apoiar
deduplicação, mas a política deve registrar como colisões e referências foram
tratadas. O texto deve manter proveniência por `id`, `domain_id`, `url_md5`,
`text_md5`, arquivo-fonte e offset; a URL bruta pode ficar somente em um
manifesto local protegido, fora do texto de treino.

Antes de fixar uma fração, ainda é necessário medir em um passe sequencial:

- contagem total por `status`;
- registros com texto e tokens após o filtro;
- duplicação por `text_md5` e, se necessário, hash recalculado;
- distribuição de bytes e tokens;
- qualidade PT-BR, boilerplate e conteúdo inadequado;
- cobertura por domínio sem expor URLs nos relatórios.

## Não é necessário reler tudo em cada execução

A preparação recomendada é:

1. registrar tamanho, mtime e um fingerprint calculado no host do arquivo;
2. fazer uma única passagem sequencial com contagens e filtros configuráveis;
3. produzir um manifesto agregado e shards derivados pequenos, preservando
   proveniência e contagem de tokens;
4. salvar cursor por byte apenas durante essa preparação, sempre no início da
   próxima linha completa;
5. nas execuções de treino, consumir somente os shards derivados até o orçamento
   configurado, sem reabrir o TSV de 56 GB;
6. invalidar os derivados se fingerprint ou configuração mudarem.

O TSV descompactado permite retomar por byte durante a primeira preparação. O
cursor deve incluir também o último `id` processado e o hash/configuração do
filtro para detectar retomadas incompatíveis. Manifestos detalhados, URLs,
textos, índices, caches e shards devem permanecer fora do Git.

## Comandos seguros e reproduzíveis

Os comandos abaixo não imprimem valores dos registros:

```sh
stat -f '%N|%z bytes|%Sm' "$PTBR_DATASET_ROOT/wacky/pages.tsv"
file "$PTBR_DATASET_ROOT/wacky/pages.tsv"

# Confirma apenas a quantidade de campos na primeira linha.
awk -F '\t' 'NR == 1 { print NF; exit }' \
  "$PTBR_DATASET_ROOT/wacky/pages.tsv"
```

Uma amostragem por offset deve abrir o arquivo em modo binário, fazer `seek`,
descartar a primeira linha parcial, impor limites de bytes/linhas e emitir
somente contagens. Não usar `cat`, `wc -l`, cópia integral ou comandos que
imprimam as colunas `url`, `title` ou `text`.

## Limitações e próxima decisão

- o total de linhas e a distribuição global de status ainda são desconhecidos;
- a amostra por offset não é estatisticamente representativa porque o arquivo
  aparenta ter ordenação correlacionada com `status`;
- não foram medidos tokens, qualidade linguística, boilerplate ou duplicação
  global;
- não foi confirmada a documentação original do schema;
- nenhuma fração foi escolhida para treino.

O próximo passo deve ser um script simples e retomável que faça a passagem
sequencial uma única vez, gere apenas métricas agregadas e, opcionalmente, um
reservoir sample protegido. Só depois dessas métricas deve-se escolher o limite
de documentos ou tokens do WackyWacky.
