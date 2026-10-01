# Reavaliação intrínseca do Adrenaline

Avaliação adicional dos três modelos existentes, sem treinamento e sem alterar
os resultados originais. O novo conjunto usa threads externas ao CPT e à
avaliação original, com controle explícito de repetição contra os blocos de treino.

## Protocolo fixado antes da inferência

- Configuração: `configs/intrinsic/adrenaline-holdout-v1.json`; seed 73129.
- Excluir todas as threads representadas no treino **ou** na avaliação original
  do Adrenaline, resolvendo os hashes dos arquivos no índice do ZIP.
- Reter até 8.192 threads por ordem SHA-256, escolher um fluxo por thread pelo
  menor hash e um bloco completo por fluxo por hash. Limite de 4 MiB por membro;
  até 4.096 blocos candidatos, antes de qualquer resultado dos modelos.
- Preservar o formatador original (`Participante N`, limpeza HTML/NFC, tokenizer
  fixado e EOS). Usar somente blocos completos de 1.024 tokens; não juntar fluxos
  ou preencher com padding. Threads sem um bloco completo ficam inelegíveis.
- Comparar exatamente os candidatos com **todos os blocos de treino preparados
  dos seis pools originais**, uma exclusão conservadora. Rejeitar o bloco inteiro
  se houver qualquer fragmento idêntico de pelo menos 32 tokens. O comparador
  não atravessa fronteiras de blocos; não usa similaridade aproximada.
- Selecionar os primeiros 256 blocos elegíveis, distintos e de 256 threads.
  São 262.144 tokens de entrada e 261.888 alvos causais. Falta de candidatos
  interrompe a execução; nenhum limiar ou cota é relaxado automaticamente.
- Nos três modelos, reproduzir primeiro os 256 blocos originais (tolerância
  absoluta 0,001 na loss). Depois medir o mesmo novo teste. Pesos FP32,
  autocast BF16, atenção eager e batch 1, como na avaliação original.
- Conferir explicitamente logits em `[:-1]` contra alvos em `[1:]`, total de
  1.023 alvos por bloco e concordância com a loss retornada pelo modelo.
- Atualizar apenas Adrenaline e a agregação global (média das seis losses,
  seguida de exponencial); copiar os outros cinco resultados sem mudanças.

O novo teste é **condicionado por esses filtros**, não uma amostra representativa
de todo o fórum. Não prova ausência de paráfrases ou exposição no pré-treino do
modelo Base, nem quantifica quanto da loss antiga se devia à memorização. O
índice ZIP usa o fingerprint original de metadados; os membros lidos têm CRC
verificado, sem calcular hash integral dos 71 GiB.

## Executar no cluster

No projeto `queroquero`, com o ambiente existente e `.env` configurado:

```bash
cd "$HOME/projects/queroquero"
source "$HOME/activate_queroquero.sh"
PREP_JOB=$(bash scripts/submit_intrinsic_holdout.sh prepare)
EVAL_JOB=$(bash scripts/submit_intrinsic_holdout.sh evaluate "$PREP_JOB")
echo "Preparação: $PREP_JOB | Avaliação: $EVAL_JOB"
```

O primeiro job solicita **8 CPUs/64 GiB/24h, sem GPU**. O segundo depende do
sucesso do primeiro e solicita **uma L40S/32 GiB/8h**, avaliando os modelos
sequencialmente. Esses tempos são limites de reserva, não durações medidas.
Requer Python 3.12.13, compilador `c++`/`g++` com C++17 e o stack já fixado
(PyTorch 2.7.1+cu118, Transformers 5.14.1). Não instala nem baixa modelos.

Logs: `logs/intrinsic-{prepare,evaluate}-<job-id>.{out,err}`. Se interromper,
submeta novamente **somente o modo interrompido**, depois de confirmar que não
há job anterior ativo. Checkpoints: indexação completa, lotes de 32 threads,
comparação de 512 candidatos e cada bloco de inferência. Uma unidade incompleta
é refeita; SIGTERM tenta preservar a unidade atual. Falta de candidatos ou
falha na reprodução original exige inspeção, não nova submissão automática.

Saída padrão: `outputs/intrinsic-holdout/<evaluation-id>/`; pode ser deslocada
por `PTBR_INTRINSIC_ROOT`. Configuração, fontes, implementação e versões formam
a identidade: alterações criam outra execução, nunca reutilizam seus checkpoints.
Dados privados (tokens, hashes e índice) ficam fora de `report/`. Somente
`report/report.json`, `report/intrinsic-results.csv` (separador `;`) e
`report/checksums.json` são agregados para inspeção. O relatório só é escrito
depois de todos os modelos e controles passarem; o relatório pareado antigo
continua intocado.

Validação/recriação dos agregados sem inferência:

```bash
bash scripts/submit_intrinsic_holdout.sh validate
bash scripts/submit_intrinsic_holdout.sh report
```

Testes sintéticos: `python -m unittest tests.test_intrinsic_holdout -v`.
