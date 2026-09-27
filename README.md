# Lab 1: Emotion Classification and Error Analysis

## My takeaway

I learned that a good accuracy score does not tell the whole story. Always guessing joy gets about 35% right without really identifying emotions. DistilBERT did better on this dataset because this version was trained for emotion classification, while BART was more like a Swiss Army knife. Descriptions helped BART, but context, capitalization, and cutting messages short can affect what a model receives. Emotions can also overlap like a Venn diagram, so I did not agree with every dataset label. I would trust DistilBERT more for this task, but I would still look at its mistakes instead of accepting every prediction.

**[Download my submission ZIP](https://github.com/StrokeOfLuck/dsa495-lab1-emotion-classification/raw/refs/heads/main/Sean-Ryan-Lab1.zip)**

The ZIP contains `Lab1.ipynb` with its saved GitHub outputs and this `README.md`. It does not include the study guide or course dataset, and it does not automatically sync with my live Colab session.

**[Open Lab 1 in Google Colab](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC)**  
[View the notebook and its saved outputs on GitHub](https://github.com/StrokeOfLuck/dsa495-lab1-emotion-classification/blob/main/Lab1.ipynb)

The Colab link opens a copy in the same Google Drive folder as `emotion.csv`. Changes saved in that Drive copy do not automatically update the GitHub notebook. The student examples and Q2 were updated in the GitHub copy on September 26, 2026; the linked Drive copy has not been synchronized with that edit.

**Before rerunning:** Choose **Runtime → Run all** from the top. The “See in Lab1” links only navigate to cells; a linked cell may fail if its setup cells have not run. If Drive mounting fails, upload `emotion.csv` when prompted.

**Student:** Sean Ryan

## My reference links

- [Trump–Iran: Emotion & Rhetoric comparison](https://strokeofluck.github.io/trump-iran-two-perspectives/): my project applying the two models to political posts.
- [Download the Lab 1 study guide](https://github.com/StrokeOfLuck/dsa495-lab1-emotion-classification/raw/refs/heads/main/study-guide.html): save the HTML file and open it in a browser. It includes code, saved outputs, figures, and my revised answers in one page. It works offline and does not require Colab or a running local server.

The guide uses the notebook outputs saved on GitHub, not the latest live Colab session. It is a reference file, separate from the required lab submission.

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

**Response:** We used 30 development messages, five per emotion, and 1,970 evaluation messages. Joy makes up 690 (35.03%) and surprise only 61 (3.10%) of the evaluation messages.

I wouldn’t call that a good model because it gives the same response every time. If it always sees joy, it isn’t really identifying the emotion in the message. It gets about 35% right because joy is common in the dataset, but that accuracy hides the fact that it misses every other emotion.

### Q2. What do the tokenizers receive?

**See in Lab1:** [Diagnostic messages I1–S2](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=73f57fce) and [token table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=71ff72e3). Compare I3 with I4, then S1 and S2 across both tokenizers; inspect `token_strings` and both length columns.

**Response:** Capitalization can mean emphasis. “I hope you’re HAPPY” sounds almost angry to me, while “I hope you’re happy” sounds softer. Lowercasing removes that emphasis cue, although the context still matters.

In I3 and I4, DistilBERT gives `HAPPY` and `happy` the same seven content tokens. BART splits `HAPPY` into `ĠH`, `APP`, and `Y`, giving nine tokens instead of seven.

In my S1, `This is NOT a positive result.`, DistilBERT lowercases `NOT`, while BART keeps it capitalized. Both have seven content tokens and nine with special tokens. In S2, `I am nervous but optimistic about leaving Raleigh.`, DistilBERT lowercases `I` and `Raleigh`, while BART keeps the capitals. Both have nine content tokens and eleven with special tokens. S1 shows emphasis and negation; S2 also has mixed emotions.

### Q3. Truncation

**See in Lab1:** [32-token truncation table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=862f59b2). Compare `retained_text` with `omitted_suffix` for each tokenizer.

**Response:** A lot can change if you cut a message short. The meaning could change with more context. At the 32-token limit, both tokenizers stop during the repeated train-ride description and leave out “I am terrified about what happens tomorrow.” That removes the clearest clue for fear. We can see the missing context here, but we haven’t shown that the prediction actually changed.

## 2. Specialized encoder classification

### Q4. Baseline and encoder results

**See in Lab1:** [Joy baseline](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=c096f034), [encoder comparison](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=e34b02b9), and [class report and confusion matrix](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=88d75891). Find surprise under `recall` and `support`.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |

**Response:** It sounds like the model has a weak area in identifying surprise, so that is a limitation to understand. It beats the joy baseline on accuracy and macro-F1, but it only catches 46 of 61 surprise messages—about 75% recall, its lowest. The overall accuracy does not tell the whole story.

### Q5. Three encoder errors

**See in Lab1:** [High-score encoder errors](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=80cf4521). Read the full text of examples `01314`, `01377`, and `01270`, alongside their reference labels and scores.

| Example ID | Reference label | Prediction | Model score | Brief observation |
|---|---|---|---:|---|
| emotion_test_01314 | surprise | fear | 0.9988 | “feel strange” is underspecified. |
| emotion_test_01377 | love | joy | 0.9986 | “overjoyed” and “beloved friends” suggest two emotions. |
| emotion_test_01270 | joy | sadness | 0.9978 | “very saddened” contradicts the reference label. |

**Response:** I think some of the terminology overlaps. Emotions can be more like a Venn diagram than something absolute. “Overjoyed” and “beloved friends” could fit joy and love. “I feel strange about it” needs more context before I would choose surprise or fear. “Very saddened” does not read as joy to me, so I would question that dataset label, though more context might help. A high model score alone does not settle which interpretation is right.

## 3. Zero-shot classification

### Q6. Label wording

**See in Lab1:** [Development comparison and changed predictions](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=927fff42). Check macro-F1, the selected formulation, and example `00332`.

| Candidate-label formulation | Accuracy | Macro-F1 |
|---|---:|---:|
| A: emotion names | 0.5000 | 0.4644 |
| B: expanded descriptions | 0.5667 | 0.5523 |

**Response:** I think the descriptions help because context helps a lot. “Fear or anxiety” gives more detail than just “fear.” B was selected because it had the higher development macro-F1 in the table. For “feel humiliated,” it changed surprise to sadness and matched the reference. That helped here, but 30 messages is too small a test to say descriptions always work better.

### Q7. Final model comparison

**See in Lab1:** [Three-method evaluation table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=f4eb1ebb) and [BART class report and confusion matrix](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=2fb8935b). Compare accuracy, macro-F1, and inference seconds on the same 1,970 messages.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |
| BART zero-shot classifier | 0.5365 | 0.4795 | 1085.14 |

**Response:** The way I think of it is that this DistilBERT is specific to the task, while this BART is more like a Swiss Army knife. DistilBERT did better on accuracy and macro-F1 and was much faster in this test. Both beat always guessing joy. That does not mean DistilBERT is better for every task. These are different checkpoints and ways of classifying text, so this is not just a test of their architectures. The times shown are from the saved CPU run and exclude model loading.

### Q8. Four model disagreements

**See in Lab1:** [Disagreement counts and four example rows](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=59bd3431). Read the text and `outcome` for `00002`, `00072`, `00098`, and `00004`.

| Example ID | Reference | DistilBERT | BART | Who is correct? |
|---|---|---|---|---|
| emotion_test_00002 | sadness | sadness | love | DistilBERT |
| emotion_test_00072 | surprise | fear | surprise | BART |
| emotion_test_00098 | anger | fear | sadness | Neither |
| emotion_test_00004 | sadness | sadness | surprise | DistilBERT |

**Response:** For `00002`, I think love fits better because when you love someone, you do not want them to hurt. “I never make her separate from me” supports BART's love prediction. “Ashamed” could suggest sadness, which DistilBERT chose, but that is not how I read the message overall.

For `00098`, I think the sentence is a combination of emotions. “My heart is tortured” could suggest sadness through emotional pain, while “what I have done” could suggest fear about consequences. Those are possible readings of BART's sadness and DistilBERT's fear predictions, not proof of what caused them. It is the Venn diagram problem again, and I would need more context to choose one.

For `00072`, “feels weird” does not really sound like fear to me. I am not sure surprise is right either, but DistilBERT seems off here.

For `00004`, maybe surprise, but I am not sure from that short phrase.

The table scores correctness against the dataset labels. My personal reading does not always agree with those labels.

### Q9. Recommendation and limitations

**See in Lab1:** [Final comparison](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=f4eb1ebb), [encoder errors](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=80cf4521), and [disagreement outcomes](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=59bd3431). Use the metrics together with the debatable example texts; also revisit the [30-message development table](https://colab.research.google.com/drive/1EMk1gvCePZZHKLl1fpkrFrfNPGG5FWeC#scrollTo=927fff42) for the selection limit.

**Response:** Given the results, I would trust DistilBERT more for this task: its accuracy was 92.44% compared with BART's 53.65%, and its macro F1 was 0.8803 compared with 0.4795. It also ran faster in the saved CPU run. I would still remember the message labeled joy that said “very saddened,” because matching the dataset does not always match how I read the text. One limitation is that single labels do not capture how emotions can overlap or how missing context changes a message. Another is that results on this dataset do not establish how either model would perform on different kinds of text.

## AI-use statement

I used ChatGPT (Codex) to complete the notebook's student code blocks, execute the analysis, and draft the interpretations in this README. Its output was checked against the supplied dataset, the notebook's displayed metrics and error rows, and the messages quoted above. I supplied replacement diagnostic sentences about a non-positive result and feeling nervous but optimistic about leaving Raleigh. Codex corrected spelling and punctuation and suggested capitalizing NOT to make the capitalization feature explicit. Codex reran the example and token-table cells using the pinned tokenizer revisions and updated Q2 to match. The existing model-evaluation outputs were retained; the full notebook was not rerun for this wording change.

For Q1 through Q9, I discussed my interpretations with Codex, which helped edit my wording and add supporting details from the results.

For the final revision, I explained why I read the first Q8 message as love and the second as a combination of emotions. Codex added possible textual support for the alternative model labels and the saved comparison numbers to Q9. Codex checked these additions against the notebook's example texts and saved results, and verified that the submission ZIP contains the current notebook and README. This revision did not rerun model inference.
