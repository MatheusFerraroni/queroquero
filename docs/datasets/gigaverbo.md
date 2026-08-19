# GigaVerbo-v2

Pré-ativação inspecionada em 2026-08-19 apenas por metadados oficiais. Nenhum
registro individual foi acessado.

## Estado

Continua desabilitado. A fonte candidata é:

| Campo | Valor |
| --- | --- |
| dataset | `Polygl0t/gigaverbo-v2` |
| revisão observada | `b39dfa703102a20dc609ed6e7aaae22e8e3a233f` |
| configuração/split | `default` / `train` |
| acesso | público, streaming |
| licença agregada | `other` |

| Configuração | Linhas | Bytes | Tokens declarados | Shards observados |
| --- | ---: | ---: | ---: | ---: |
| `default` | 372.108.576 | 834.750.093.054 | 317.688.116.144 | 224 |
| `excluded` | 2.892.095 | 7.894.131.263 | 2.987.598.133 | 23 |

`excluded` não foi selecionado. O card ainda menciona 103 shards `default`,
mas a revisão observada contém 224; a implementação deve resolver os arquivos
da revisão pinada.

## Schema e seleção

Campos: `text`, `id`, `source`, `subset`, `token_count`, `edu_score`,
`edu_int_score`, `toxic_score` e `toxic_int_score`.

Regras candidatas:

- usar somente `default/train` por streaming;
- filtrar `edu_int_score >= 4`;
- exigir budget finito por documentos ou tokens;
- pin revision, seed, ordem e buffer;
- retokenizar com o tokenizer do `Tucano2-0.6B-Base`.

`token_count` usa outro tokenizer e serve apenas como estimativa. Uma
estatística parcial de 1.806.353 linhas encontrou 3,36% com score `>= 4`; esse
valor não representa o corpus completo.

## Configuração candidata

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
```

A ativação exige orçamento, revisão de licenças, manifesto dos shards, métricas,
retomada e testes sintéticos. O protótipo atual ainda não implementa esses
itens.

O corpus combina fontes com licenças e termos diferentes. Dados e pesos
derivados permanecem `internal_research_only`.

## Referências

- [Dataset card](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2)
- [Revisão observada](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2/tree/b39dfa703102a20dc609ed6e7aaae22e8e3a233f)
- [Tamanhos oficiais](https://datasets-server.huggingface.co/size?dataset=Polygl0t%2Fgigaverbo-v2)
