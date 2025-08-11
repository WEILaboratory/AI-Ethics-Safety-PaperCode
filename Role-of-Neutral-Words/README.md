# README

## collect sentences

```bash
cd main/scripts
bash collect_sentences_parallel.sh bert news_commentary.txt gender_brief
```

## debias gender_brief-neutral

```bash
bash debias_gender-neutral.sh
```

## evaluation on Stereoset

```bash
bash evaluate_gender-neutral.sh
```

## collect sentences for scm-neutral

```bash
cd main/scripts
bash collect_sentences_parallel.sh bert news_commentary.txt scm
```

## debias scm-neutral

```bash
bash debias_scm-neutral.sh
```

## evaluation on Stereoset

```bash
bash evaluate_scm-neutral.sh
```

## collect sentences for group-specific neutral words

```bash
cd main/scripts
bash collect_sentences_parallel_group-specific_neutral.sh bert news_commentary.txt scm
```

```bash
cd main/scripts
bash collect_sentences_parallel_group-specific_neutral.sh bert news_commentary.txt gender_brief
```

```bash
cd main/scripts
bash collect_sentences_parallel_group-specific_neutral.sh bert news_commentary.txt religion
```

## collect sentences for gender-scm

```bash
cd main/scripts
bash collect_sentences_parallel.sh bert news_commentary.txt gender_brief-scm
```
