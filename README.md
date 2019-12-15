# Episodic Memory in Lifelong Language Learning
Code for the paper: `Episodic Memory in Lifelong Language Learning`[(Arxiv:1905.12926)](https://arxiv.org/pdf/1906.01076v3.pdf) for the text classification setup.
## Introduction
The ability to continuously learn and accumulate knowledge throughout a lifetime and reuse it effectively to adapt to a new problem quickly is a hallmark of general intelligence. State-of-the-art machine learning models work well on a single dataset given enough training examples, but they often fail to isolate and reuse previously acquired knowledge when the data distribution shifts (e.g., when presented with a new dataset)—a phenomenon known as catastrophic forgetting. In this paper, the authors  introduce a lifelong language learning setup where a model needs to learn from a stream of text examples without any dataset identifier. Specificaly, they propose an episodic memory model that performs sparse experience replay and local adaptation to mitigate catastrophic forgetting in this setup. Experiments on the text classification and question answering tasks demonstrate that the episodic memory module is a crucial building block of general linguistic intelligence.

## Model
Main components of the model
### Example Encoder
Text classification : x<sub>t</sub> is a document to be classified; BERT produces a vector representation of each token in x<sub>t</sub>, which includes a special beginning-of-document symbol CLS as x<sub>t,0</sub>.   
Question answering: x<sub>t</sub> is a concatenation of a context paragraph x<sub>t</sub><sup>context</sup> and a question x<sub>t</sub><sup>question</sup> separated by a special separator symbol SEP.
### Task Decoder
*Text classification:* following the original BERT model, select the representation of the first token x<sub>t,0</sub> from BERT (i.e., the special beginning-of-document symbol) and add a linear transformation and a softmax layer to predict the class of x<sub>t</sub>.
