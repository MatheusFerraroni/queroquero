# BrWaC-CLEAN: exploração inicial

Data da inspeção: 2026-08-18. O arquivo está em um mount SSHFS somente leitura.
Esta nota registra somente metadados, estatísticas agregadas e características de
formato. Nenhum texto, URL, identificador de pessoa ou outro valor do corpus foi
incluído no repositório; o ZIP não foi extraído integralmente.

## Inventário confirmado

| Arquivo | Tamanho no mount | Entradas | Conteúdo observado |
| --- | ---: | ---: | --- |
| `BrWac-CLEAN/BrWac.zip` | 5.837.490.062 bytes | 3.063.730 | `data/`, documentos `.txt` e `names.tsv` |

O cabeçalho do ZIP informa 3.063.730 entradas. A enumeração do diretório central
classificou-as como:

- `data/`: 1 diretório, armazenado sem compressão;
- `data/<identificador>.txt`: 3.063.728 arquivos;
- `names.tsv`: 1 arquivo.

O `zipinfo -t` reportou 12.574.567.129 bytes descompactados e 5.341.755.770
bytes compactados, uma redução agregada indicada pelo utilitário como 57,5%.
Subtraindo `names.tsv`, os documentos `.txt` representam aproximadamente
12.091.757.071 bytes descompactados e 5.168.402.429 bytes compactados. Esses
valores são totais do diretório central, não uma leitura do conteúdo.

O diretório central tem 269.989.229 bytes e começa no offset 5.567.500.735 do
ZIP. Isso explica por que uma simples enumeração pode ser custosa quando o
arquivo é acessado por SSHFS, mesmo sem descompactar os documentos.

## Documentos `data/*.txt`

Os membros consultados são arquivos de texto simples, com nomes numéricos e
compressão DEFLATE (`defN`). Uma amostra determinística limitada de 11 membros
teve:

| Medida da amostra | Resultado |
| --- | ---: |
| tamanho descompactado | 379–20.758 bytes |
| linhas por membro | 2–108 |
| encoding decodificado | UTF-8 em todos os membros |
| tags HTML/XML reconhecíveis | 0 na amostra |
| tabs | 0 na amostra |
| estrutura JSON | não observada |

Os arquivos contêm várias linhas, sem linhas vazias consecutivas na amostra.
Isso sugere uma unidade documental por membro com segmentação interna por linhas,
mas não confirma se cada linha é uma sentença, parágrafo ou outro segmento. O
schema semântico e a presença de boilerplate, duplicação, idiomas mistos ou
conteúdo inadequado ainda precisam ser medidos sem registrar os valores.

O maior tamanho acima é apenas o máximo da amostra. A distribuição completa não
foi calculada para evitar uma varredura longa do diretório central pelo mount.
Também não foi feita uma leitura do maior membro do arquivo; um pipeline futuro
deve impor limite de bytes por documento e tratar outliers sem carregá-los todos
na memória.

## `names.tsv`

O membro tem 482.810.058 bytes descompactados e 173.353.341 bytes compactados,
também com DEFLATE. Uma amostra limitada de 1 MiB decodificou como UTF-8 e
apresentou registros separados por newline e três campos separados por tab. A
amostra teve 6.613 linhas completas, sem campos vazios ou aspas CSV observadas;
um registro final parcial foi separado por estar no limite da amostra.

Os nomes e valores dos campos não foram impressos. Não foi confirmado se existe
linha de cabeçalho nem qual é a semântica de cada coluna. Pelo tamanho, este
arquivo é um índice/metadado relevante e não deve ser reprocessado como texto de
treino antes de uma decisão explícita sobre a proveniência e a necessidade de
seus campos.

## Não é necessário reler o ZIP em cada execução

O arquivo deve ser tratado como entrada versionada e preparada uma vez:

1. registrar tamanho, mtime e, quando houver uma janela de I/O adequada, um
   fingerprint do ZIP;
2. gerar no host que armazena o arquivo um manifesto do diretório central com
   `member_path`, extensão, tamanho compactado/descompactado, CRC, método e
   grupo (`data` ou índice);
3. selecionar uma amostra determinística por tamanho/quantil e validar UTF-8,
   segmentação, duplicação e regras de limpeza;
4. processar somente os membros selecionados para shards derivados, mantendo
   `archive` e `member_path` como proveniência;
5. salvar um cursor por membro/shard para retomar após interrupções.

O manifesto detalhado, caches, documentos derivados e shards devem ficar fora do
Git. O repositório deve conter apenas este tipo de relatório, schemas,
estatísticas agregadas e a configuração que reproduz a seleção. O fingerprint
deve ser comparado antes de reutilizar um manifesto; se tamanho/mtime mudarem,
o manifesto deve ser invalidado e regenerado.

## Comandos seguros e reproduzíveis

Os comandos abaixo consultam o diretório central ou limitam a leitura de um
membro. Eles não devem ser substituídos por extração integral:

```sh
stat -f '%N|%z bytes|%Sm' "$PTBR_DATASET_ROOT/BrWac-CLEAN/BrWac.zip"
zipinfo -h "$PTBR_DATASET_ROOT/BrWac-CLEAN/BrWac.zip"
zipinfo -t "$PTBR_DATASET_ROOT/BrWac-CLEAN/BrWac.zip"

# Conferir somente a estrutura dos primeiros nomes; não imprime conteúdo.
zipinfo -1 "$PTBR_DATASET_ROOT/BrWac-CLEAN/BrWac.zip" | head -40

# Consultar metadados de membros escolhidos.
zipinfo -l "$PTBR_DATASET_ROOT/BrWac-CLEAN/BrWac.zip" \
  data/2730063.txt names.tsv
```

Para validar encoding e segmentação, deve-se abrir alguns membros pequenos com
`ZipFile.open()` ou `unzip -p`, limitar a leitura a uma quantidade fixa de bytes
e emitir apenas contagens, tipos e erros agregados. No macOS inspecionado não há
`timeout`/`gtimeout`; uma nova operação potencialmente longa deve usar um
wrapper Python com alarme/timeout explícito ou ser executada no host do ZIP.

## Limitações e próximo passo

- não foi medida a distribuição completa dos tamanhos dos documentos;
- não foi confirmado o schema semântico das linhas dos `.txt`;
- não foram calculadas deduplicação, proporção de boilerplate ou qualidade
  linguística;
- não foi confirmado o cabeçalho e o papel das três colunas de `names.tsv`;
- não foi feita nenhuma limpeza nem decisão sobre fração para treino.

O próximo passo recomendado é uma amostra bounded e reprodutível de documentos
pequenos, médios e grandes, junto com poucos registros de `names.tsv` usados
somente para verificar a ligação por identificador. Essa fase deve medir tamanho
de texto útil, segmentação, duplicação aproximada e custo de sanitização, sempre
descartando os valores logo após a medição. Só depois deve-se decidir se o
treino usará documentos completos, segmentos limitados ou uma fração amostrada.
