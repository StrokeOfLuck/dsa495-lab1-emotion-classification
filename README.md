# Lab 1: Emotion Classification and Error Analysis

**[Open Lab 1 in Google Colab](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC)**  
[View the notebook and its saved outputs on GitHub](https://github.com/StrokeOfLuck/dsa495-lab1-emotion-classification/blob/main/Lab1.ipynb)

The Colab link opens a copy in the same Google Drive folder as `emotion.csv`. Changes saved in that Drive copy do not automatically update the GitHub notebook. The student examples and Q2 were updated in the GitHub copy on September 26, 2026; the linked Drive copy has not been synchronized with that edit.

**Before rerunning:** Choose **Runtime → Run all** from the top. The “See in Lab1” links only navigate to cells; a linked cell may fail if its setup cells have not run. If Drive mounting fails, upload `emotion.csv` when prompted.

**Student:** Sean Ryan

## Purpose

This task assigns one of six emotion labels to each message in the supplied dataset. Comparing a constant joy prediction, a DistilBERT checkpoint trained for emotion classification, and a BART natural-language-inference checkpoint used with zero-shot candidate labels shows how task-specific training and label wording affect performance on the same held-out messages.

## Files and rerun instructions

- `Lab1.ipynb`: code, displayed tables and figures for the split, tokenizer inspection, model predictions, evaluation, and error analysis.
- `README.md`: written responses and reproducibility instructions.
- No additional submission files. The course dataset is already provided separately.

To run the analysis:

1. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. When the setup cell asks to mount Google Drive, approve it. The file is in `MyDrive/Text analysis for Data Science/Lab 1`. If mounting fails, the same cell opens an upload dialog; select `emotion.csv` from your computer. You can download it from the course Drive folder first if needed.
3. Run all cells from top to bottom. The first cell installs the pinned `transformers` version; the first model run downloads the two pinned checkpoints. Keep the random seed and split unchanged to reproduce the comparisons. In a local run, place `emotion.csv` beside the notebook instead.

## 1. Data and tokenization

### Q1. Development and evaluation data

**See in Lab1:** [Split-count table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=c433329a). Read the `development`, `evaluation`, and `evaluation_percent` columns, especially joy and surprise.

**Response:** We set aside 30 development messages, five for each emotion, and used the remaining 1,970 for evaluation. Joy has 690 evaluation messages (35.03%), while surprise has only 61 (3.10%).

I wouldn’t call that a good model because it gives the same response every time. If it always sees joy, it isn’t really identifying the emotion in the message. It gets about 35% right because joy is common in the dataset, but that accuracy hides the fact that it misses every other emotion.

That is why I would also look at macro-F1 and recall for each emotion instead of relying only on accuracy.

### Q2. What do the tokenizers receive?

**See in Lab1:** [Diagnostic messages I1–S2](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=73f57fce) and [token table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=71ff72e3). Compare I3 with I4, then S1 and S2 across both tokenizers; inspect `token_strings` and both length columns.

**Response:** Capitalization can mean emphasis, and emphasizing a word can change how a sentence comes across in a positive or negative way. For example, “I hope you’re HAPPY” sounds almost angry or sarcastic to me, while “I hope you’re happy” sounds softer. The surrounding context still matters, but lowercasing everything removes that capitalization cue. Keeping the capitals does not automatically mean the model understands the tone correctly.

In the lab’s I3 and I4 examples, DistilBERT turns `HAPPY` into `happy`, so both sentences have the same seven content tokens. BART keeps the capitalization and splits `HAPPY` into `ĠH`, `APP`, and `Y`, giving it nine content tokens compared with seven for the lowercase version.

My S1 example is `This is NOT a positive result.` DistilBERT lowercases `This` and `NOT`, while BART keeps `This` and `ĠNOT`. Both have seven content tokens and nine with special tokens. The word `not` is still there in DistilBERT; what gets lost is the extra emphasis from the capitals. In S2, `I am nervous but optimistic about leaving Raleigh.`, DistilBERT lowercases `I` and `Raleigh`, while BART keeps their capitalization. Both have nine content tokens and eleven with special tokens, and both keep `nervous` and `optimistic` as single tokens. S1 shows capitalization and negation, while S2 also gives an example of mixed emotions.

### Q3. Truncation

**See in Lab1:** [32-token truncation table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=862f59b2). Compare `retained_text` with `omitted_suffix` for each tokenizer.

**Response:** A lot can change if you cut a message short. The meaning could change with more context, so the part that gets left out might be the most important part. In this example, both tokenizers stop partway through the second repetition of the ordinary train-ride description at the artificial 32-token limit. They leave out the entire ending: “Despite the ordinary journey, I am terrified about what happens tomorrow.” Without that ending, the model loses the clearest clue for fear and only sees the earlier description of the trip. That could change its interpretation, although this example shows what text was removed, not an actual change in the model’s prediction.

## 2. Specialized encoder classification

### Q4. Baseline and encoder results

**See in Lab1:** [Joy baseline](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=c096f034), [encoder comparison](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=e34b02b9), and [class report and confusion matrix](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=88d75891). Find surprise under `recall` and `support`.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |

**Response:** It sounds like the model has a weak area when it comes to identifying surprise, and that is a limitation we need to understand. DistilBERT does much better than always guessing joy: its accuracy is 0.9244 compared with 0.3503, and its macro-F1 is 0.8803 compared with 0.0865. But the strong overall result does not mean it handles every emotion equally well. Surprise has its lowest recall, at 0.7541: it correctly identifies 46 of the 61 surprise messages and misses 15. I would look at the results for each emotion instead of letting the overall accuracy hide that weak area. These results show a limitation on this dataset, not proof that the model always struggles with surprise in every setting.

### Q5. Three encoder errors

**See in Lab1:** [High-score encoder errors](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=80cf4521). Read the full text of examples `01314`, `01377`, and `01270`, alongside their reference labels and scores.

| Example ID | Reference label | Prediction | Model score | Brief observation |
|---|---|---|---:|---|
| emotion_test_01314 | surprise | fear | 0.9988 | “feel strange” is underspecified. |
| emotion_test_01377 | love | joy | 0.9986 | “overjoyed” and “beloved friends” suggest two emotions. |
| emotion_test_01270 | joy | sadness | 0.9978 | “very saddened” contradicts the reference label. |

**Response:** I think some of the terminology overlaps. Emotions can be more like a Venn diagram than something absolute, so forcing a message into one category can miss that overlap. In `emotion_test_01377`, “overjoyed” sounds like joy, while “beloved friends” also brings in love. I can see why the model and the dataset picked different labels there.

For `emotion_test_01314`, “I feel strange about it” does not give me enough context to confidently choose surprise or fear. I would want to know more about what the person meant. In `emotion_test_01270`, I would probably question the dataset’s joy label because “very saddened” does not read as joy to me. More context might help, but based on the text we have, sadness seems more reasonable.

These examples make me cautious about calling every disagreement a clear model mistake. The reference label could be questionable, the wording could be vague, or more than one emotion could fit. At the same time, the model’s scores near 1 do not prove it is right; we still need to read the message and consider the context.

## 3. Zero-shot classification

### Q6. Label wording

**See in Lab1:** [Development comparison and changed predictions](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=927fff42). Check macro-F1, the selected formulation, and example `00332`.

| Candidate-label formulation | Accuracy | Macro-F1 |
|---|---:|---:|
| A: emotion names | 0.5000 | 0.4644 |
| B: expanded descriptions | 0.5667 | 0.5523 |

**Response:** I think the descriptions help because context helps a lot. Giving BART “fear or anxiety” instead of just “fear” gives it a fuller idea of what we mean by that label. It adds detail to the label, even though the message itself stays the same.

In this development test, the expanded descriptions had a macro-F1 of 0.5523 compared with 0.4644 for the emotion names alone. That is why the rule of choosing the higher development macro-F1 selected formulation B before evaluation. For `emotion_test_00332`, the message with “feel humiliated” changed from surprise to sadness, which matched the reference label.

The descriptions helped in this test, but there were only 30 development messages, five per emotion. A few changed predictions could make a big difference, so this does not show that longer descriptions will always work better on new messages.

### Q7. Final model comparison

**See in Lab1:** [Three-method evaluation table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=f4eb1ebb) and [BART class report and confusion matrix](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=2fb8935b). Compare accuracy, macro-F1, and inference seconds on the same 1,970 messages.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |
| BART zero-shot classifier | 0.5365 | 0.4795 | 1085.14 |

**Response:** DistilBERT exceeds BART by 0.3878 in accuracy and 0.4008 in macro-F1 on these messages. BART still exceeds the constant baseline on both metrics. These checkpoints differ in training, task, size, and inference procedure, so this is an operational comparison of the supplied approaches rather than an isolated test of encoder versus encoder-decoder architecture; timings are from this CPU run and exclude model loading.

### Q8. Four model disagreements

**See in Lab1:** [Disagreement counts and four example rows](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=59bd3431). Read the text and `outcome` for `00002`, `00072`, `00098`, and `00004`.

| Example ID | Reference | DistilBERT | BART | Who is correct? |
|---|---|---|---|---|
| emotion_test_00002 | sadness | sadness | love | DistilBERT |
| emotion_test_00072 | surprise | fear | surprise | BART |
| emotion_test_00098 | anger | fear | sadness | Neither |
| emotion_test_00004 | sadness | sadness | surprise | DistilBERT |

**Response:** In `emotion_test_00072`, “feels weird” offers BART a plausible surprise cue, while unfamiliar bodily coordination may have led DistilBERT toward fear; the reference favors BART, though the message does not explicitly name an emotion. In `emotion_test_00098`, “heart is tortured by what i have done” supports DistilBERT's fear and BART's sadness or guilt readings, but it does not plainly express the reference anger. The “ashamed” wording in `emotion_test_00002` supports sadness despite its relational context, while the brief “vain” message in `emotion_test_00004` leaves little context. Across all disagreements, DistilBERT alone is correct in 812, BART alone in 48, and neither in 44; these cases warrant reading the text, not merely counting labels.

### Q9. Recommendation and limitations

**See in Lab1:** [Final comparison](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=f4eb1ebb), [encoder errors](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=80cf4521), and [disagreement outcomes](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=59bd3431). Use the metrics together with the debatable example texts; also revisit the [30-message development table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=927fff42) for the selection limit.

**Response:** I would use the specialized DistilBERT classifier for this fixed six-emotion task: its 0.9244 accuracy and 0.8803 macro-F1 exceed BART's 0.5365 and 0.4795, and it ran in 22.82 rather than 1085.14 seconds on this CPU. The 812 versus 48 single-model wins reinforce that choice, although BART recognizes the “feels weird” surprise example that DistilBERT misses. The labeled data contain debatable single-emotion references, such as joy for “very saddened”; these metrics cannot establish correctness for nuanced or multiple emotions. The evaluation uses one source test split and one tiny development selection, so it does not establish performance on other domains, new labels, or alternative prompt and checkpoint choices. Model scores were not calibrated here, and the runtime comparison excludes checkpoint loading and depends on hardware.

## AI-use statement

I used ChatGPT (Codex) to complete the notebook's student code blocks, execute the analysis, and draft the interpretations in this README. Its output was checked against the supplied dataset, the notebook's displayed metrics and error rows, and the messages quoted above. I supplied replacement diagnostic sentences about a non-positive result and feeling nervous but optimistic about leaving Raleigh. Codex corrected spelling and punctuation and suggested capitalizing NOT to make the capitalization feature explicit. Codex reran the example and token-table cells using the pinned tokenizer revisions and updated Q2 to match. The existing model-evaluation outputs were retained; the full notebook was not rerun for this wording change.
