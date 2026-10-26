# Next-Speaker Prediction in Multi-Party Meetings

Predicting **who will speak next** in a four-person meeting, using a hierarchical **BiLSTM** over the dialogue history. Next-speaker prediction matters for real-time conversational agents that take part in group conversations and must decide when to speak.

Built for the ECS7033P Conversational Agents & Dialogue Systems module at Queen Mary University of London (2026).

## Data
The [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) annotations (University of Edinburgh). The notebook downloads them automatically.
- Word-level transcripts are merged into speaker turns (same speaker, gap ≤ 0.5 s).
- Only meetings with exactly four speakers are kept: 147 meetings, ~197k turns.
- Each example has the current utterance plus the **8 previous turns**, labelled with the next speaker (A–D).
- The split is **by meeting** (80/10/10), so no meeting appears in both training and test: 161k / 13k / 22k examples.

## Model
```
each utterance → word embeddings → BiLSTM utterance encoder
              (+ previous-speaker embedding)
8 history turns + current turn → context BiLSTM
              (+ current-speaker embedding)
              → MLP classifier → next speaker (4 classes)
```
Two switches control whether current-speaker and previous-speaker information are used, which makes the ablation study straightforward. Training uses Adam, gradient clipping, a learning-rate scheduler on dev macro-F1 and early stopping.

## Results (test set)
| Configuration | Accuracy | Macro F1 |
|---|---|---|
| Current + previous speakers | 71.16% | 0.710 |
| Previous speakers only | 68.54% | 0.684 |
| Current speaker only | 33.84% | 0.328 |
| No speaker information | 27.31% | 0.262 |

The ablation shows that **who spoke in the previous turns is by far the strongest signal**: without it, accuracy drops to around chance level (four classes, roughly 25%), even when the model still sees the words.

*Note:* these values were recorded from separate training runs and collected into a table at the end of the notebook. The training run saved in the notebook (both speaker features on) reports 68.5%.

## How to run
```bash
pip install -r requirements.txt
```
Then run the notebook top to bottom. A GPU is recommended.

## Tech stack
Python · PyTorch · pandas · scikit-learn

---
**Author:** Sude Kumbasar
