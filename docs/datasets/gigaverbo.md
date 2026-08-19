# GigaVerbo-v2: exploração e pré-ativação

Data da inspeção: 2026-08-19. Esta exploração consultou somente o card, o
repositório e as APIs oficiais do Hugging Face. Nenhum registro individual foi
baixado ou impresso; textos, IDs e valores de proveniência dos documentos não
foram copiados para o repositório.

## Estado da decisão

O GigaVerbo-v2 continua **desabilitado**. Este documento prepara uma decisão
futura de ativação, mas não a substitui. Uma ativação deverá ser registrada de
forma explícita no README e em uma configuração executável e versionada.

Se ativado, o contrato já definido pelo projeto exige:

- acesso exclusivamente por streaming, sem download do corpus completo;
- configuração `default` e split `train`;
- filtro `edu_int_score >= 4`;
- limite obrigatório e configurável por documentos ou tokens;
- revisão imutável do dataset, seed e parâmetros de seleção registrados;
- preservação de proveniência e produção de shards derivados retomáveis;
- permanência dos dados, caches, manifests detalhados e shards fora do Git.

A configuração `excluded` não foi selecionada. Ela contém documentos separados
no fim da filtragem upstream, incluindo material curto ou com toxicidade alta,
e não deve entrar por engano no continual pretraining.

## Identidade e revisão observada

| Campo | Valor observado |
| --- | --- |
| Dataset | `Polygl0t/gigaverbo-v2` |
| Revisão do repositório | `b39dfa703102a20dc609ed6e7aaae22e8e3a233f` |
| Configuração candidata | `default` |
| Split | `train` |
| Acesso | público, não gated |
| Formato | Parquet |
| Idioma declarado | português |
| Licença agregada | `other`; não há uma licença única para o corpus combinado |

A revisão acima é a observada durante esta inspeção. Ela é uma candidata para a
configuração inicial, não uma autorização de uso. Antes da ativação, deve-se
confirmar que continua disponível e registrar no manifesto a lista de arquivos
resolvida nessa revisão.

## Inventário oficial

Os metadados oficiais informam:

| Configuração | Split | Linhas | Bytes | Tokens declarados no card | Shards na revisão observada |
| --- | --- | ---: | ---: | ---: | ---: |
| `default` | `train` | 372.108.576 | 834.750.093.054 | 317.688.116.144 | 224 |
| `excluded` | `train` | 2.892.095 | 7.894.131.263 | 2.987.598.133 | 23 |
| **Total** | — | **375.000.671** | **842.644.224.317** | **320.675.714.277** | **247** |

Na revisão observada, os arquivos seguem os padrões
`default/train-00000-of-00224.parquet` a
`default/train-00223-of-00224.parquet` e
`excluded/train-00000-of-00023.parquet` a
`excluded/train-00022-of-00023.parquet`.

Há uma divergência documental apenas na quantidade de arquivos `default`: a
tabela narrativa do card ainda registra 103, enquanto a árvore do repositório e
a API de Parquet registram 224 na revisão acima. As quantidades de linhas e
bytes coincidem. O código futuro não deve fixar a contagem narrativa; deve
resolver e manifestar os shards da revisão pinada.

## Schema confirmado por metadados

A configuração `default` declara nove campos:

| Campo | Tipo | Uso previsto neste projeto |
| --- | --- | --- |
| `text` | `string` | texto candidato ao pretraining |
| `id` | `string` | identificador MD5 fornecido pelo dataset; proveniência e deduplicação |
| `source` | `string` | fonte upstream; metadado protegido, nunca valor de log |
| `subset` | `string` | subconjunto upstream; estratificação e proveniência |
| `token_count` | `int64` | estimativa upstream, não orçamento final deste projeto |
| `edu_score` | `float64` | score educacional contínuo do classificador upstream |
| `edu_int_score` | `int64` | score educacional arredondado usado pelo filtro do projeto |
| `toxic_score` | `float64` | score de toxicidade contínuo upstream |
| `toxic_int_score` | `int64` | score de toxicidade arredondado upstream |

O card informa que `id` é um MD5 do documento e que `token_count` foi calculado
com o tokenizer Tucano-2b4. Portanto, `token_count` pode apoiar estimativas de
I/O, mas não pode definir o orçamento final nem validar sequências deste
projeto. Todo texto selecionado deve ser retokenizado com o tokenizer original
do `Polygl0t/Tucano2-0.6B-Base` na revisão pinada pelo projeto.

O conjunto `default` já passou, segundo o produtor, por extração, identificação
de idioma, filtros heurísticos, formatação, remoção de PII, deduplicação MinHash
e classificadores de qualidade educacional e toxicidade. Essas etapas são
evidência upstream, não garantia de ausência de PII, conteúdo inadequado,
duplicatas ou contaminação de idioma. Valores de `text`, `id` e `source` não
devem aparecer em documentação, fixtures, logs ou testes.

## Exploração estatística limitada

O endpoint de estatísticas do Dataset Viewer marcou sua resposta como parcial e
abrangeu 1.806.353 exemplos, menos de 0,5% do `default`. Essa amostra confirmou
tipos e ausência de nulos nas colunas consultadas, mas não é representativa do
corpus completo.

| Medida parcial | Resultado observado |
| --- | ---: |
| `edu_int_score == 1` | 703.661 |
| `edu_int_score == 2` | 743.969 |
| `edu_int_score == 3` | 298.083 |
| `edu_int_score >= 4` | 60.640 (3,36%) |
| mediana de `token_count` upstream | 331 |
| média de `token_count` upstream | 605,92 |
| máximo de `token_count` upstream | 173.116 |
| classes de `source` / `subset` presentes | 3 / 2 |

O rendimento de 3,36% não deve ser extrapolado para planejar o número total de
documentos ou tokens. A ordem dos shards e a resposta parcial podem introduzir
viés. O rendimento real do filtro `edu_int_score >= 4` e a contagem com o
tokenizer do modelo devem ser medidos durante uma preparação limitada e
retomável.

O card publica um total de 118.734.942.208 tokens para seu conceito de conteúdo
educacional, mas usa `edu_int_score >= 3`. Esse total não representa o filtro
mais estrito `>= 4` adotado por este projeto.

## Licenças e uso permitido

O card usa a licença agregada `other` e orienta o consumidor a obedecer às
licenças e aos termos de cada fonte. A composição inclui Common Crawl e
datasets com combinações de ODC-By, termos do Common Crawl, CC0, CC-BY,
CC-BY-SA, Apache-2.0, MIT, CC-BY-NC, CC-BY-NC-SA e domínio público.

Consequentemente:

1. a ativação para pesquisa interna não autoriza redistribuição do corpus ou
   dos pesos derivados;
2. o manifesto deverá registrar a revisão, `source` e `subset` de forma
   agregável, sem publicar valores de registros;
3. pesos refinados continuam `internal_research_only`;
4. uma revisão de licenças, termos e permissões continua sendo condição para
   qualquer distribuição posterior.

## Configuração candidata, ainda inativa

O formato abaixo documenta os campos mínimos esperados. Ele é ilustrativo e não
é uma configuração executável enquanto orçamento, seed e buffer não forem
decididos:

```yaml
gigaverbo_v2:
  enabled: false
  dataset_id: Polygl0t/gigaverbo-v2
  revision: b39dfa703102a20dc609ed6e7aaae22e8e3a233f
  config_name: default
  split: train
  streaming: true
  min_edu_int_score: 4
  budget:
    unit: tokens
    value: <definir>
  selection:
    seed: <definir>
    shuffle_buffer_size: <definir>
  fields:
    text: text
    id: id
    source: source
    subset: subset
```

Nenhuma configuração deve aceitar `enabled: true` com orçamento ausente, zero,
negativo ou ilimitado. Uma mudança de revisão, threshold, unidade de orçamento,
seed, buffer, tokenizer ou política de truncamento deve invalidar os derivados.

## Fluxo recomendado para ativação

1. Registrar a decisão explícita, o orçamento, a seed e a política de seleção.
2. Resolver a revisão pinada e salvar, fora do Git, um manifesto dos 224 shards
   `default`, incluindo caminho, tamanho e identificador remoto disponível.
3. Abrir somente `default`/`train` com streaming e validar o schema antes de ler
   documentos.
4. Aplicar `edu_int_score >= 4`; qualquer filtro adicional de toxicidade,
   origem ou comprimento precisa ser uma decisão explícita.
5. Embaralhar de forma bounded com seed e tamanho de buffer registrados. Esse
   método é reprodutível para a mesma revisão, versão da biblioteca e ordem dos
   shards, mas não equivale a uma permutação global uniforme.
6. Retokenizar `text` com o tokenizer imutável do modelo, aplicar a política de
   sequência de 1.024 tokens e encerrar no orçamento configurado.
7. Produzir shards derivados pequenos, atômicos e retomáveis. Registrar por
   shard a revisão upstream, shard/ordinal de origem, digest agregado dos IDs,
   contagens de documentos/tokens, rejeições e hashes.
8. Emitir métricas agregadas por identificador canônico ou hash de `source`,
   por `subset` e por score, sem textos, IDs ou URLs crus, e manter
   métricas/manifests separados de checkpoints.

Para retomada eficiente, o cursor deve ser por shard upstream e ordinal do
registro, não apenas pelo total global de documentos. Reabrir um streaming e
usar somente `skip()` pode reler uma grande parte do corpus; shards upstream já
concluídos devem ser reconhecidos pelo manifesto.

## Lacunas do protótipo atual

`prototypes/gigaverbo_streaming.py` já usa streaming e o threshold correto, mas
não pode ser promovido diretamente a entrypoint. Antes disso, faltam:

- revisão pinada e configuração `default`/`train` explicitamente validadas;
- orçamento obrigatório e sem caminho ilimitado;
- seed, seleção bounded e política de ordenação;
- validação de schema e tipos;
- retokenização com o tokenizer do modelo;
- shards derivados, manifesto, métricas e retomada;
- tratamento de erros e ausência de efeitos de rede no import;
- testes que usem somente registros sintéticos sem valores dos corpora.

O protótipo deve permanecer inativo até essas condições e a decisão explícita
de ativação serem atendidas.

## Referências oficiais

- [Dataset card do GigaVerbo-v2](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2)
- [Revisão observada](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2/tree/b39dfa703102a20dc609ed6e7aaae22e8e3a233f)
- [Metadados de tamanho do Dataset Server](https://datasets-server.huggingface.co/size?dataset=Polygl0t%2Fgigaverbo-v2)
- [Tucano 2 Cool](https://huggingface.co/papers/2603.03543)
