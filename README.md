<h1 align="center">
    Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems
</h1>
<p align="center">
    <a href="https://arxiv.org/pdf/2601.05171">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv">
    </a>
    <a href="https://huggingface.co/Robot2050/MemListener">
        <img src="https://img.shields.io/badge/Huggingface-MemListener-yellow?style=flat-square&logo=huggingface">
    </a>
    <a href="https://opensource.org/license/apache-2-0">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg?logo=apache">
    </a>
</p>

|  |  |
| --- | --- |
| <img src="img/InsideOut.jpg" alt="InsideOut" width="2000"> |  **🎯 Who Should Pay Attention to Our Work?** <br> **Researchers in Long-Term Dialogue Systems and Personalization:**<br> Scholars addressing the "forgetting" and "hallucination" problems  in lifelong open-domain conversation will find the **Inside Out** framework crucial. It tackles the trade-off between finite context windows and unbounded interaction histories, offering a solution to persona inconsistency and memory noise accumulation.<br>**Engineers Focused on Efficient LLM Deployment:** <br>Practitioners seeking to reduce inference latency and computational costs without sacrificing model performance should note this work. The paper demonstrates a collaborative paradigm where a lightweight model handles memory management, allowing the larger LLM to focus on generation, thus optimizing resource allocation in latency-sensitive applications.<br> **Interdisciplinary Scientists in Cognitive AI:**<br> Researchers interested in integrating psychological theories into computational architectures will value the **PersonaTree** design. This work validates the utility of structured, hierarchically organized memory over unstructured vector retrieval for modeling human-like identity retention. |




## 📊 Quick Start

* **`infer_llm.py`**: Performs inference using only a LLM, without incorporating additional memory or structured representations.

* **`infer_llm_all.py`**: Conducts LLM-based inference by providing the model with the full dialogue history as input context.

* **`infer_personamem.py`**: Executes inference solely based on the PersonaTree.

* **`infer_memrewriter.py`**: Executes inference based on PersonaTree and PersonaTree–augmented retrieval.

* **`memtree4user.py`**: Provides example code demonstrating how to interface with a MongoDB backend and implement streaming operations for memory/tree updates.

* **`pipline_ablation.py`**: Implements an ablation-style pipeline that first extracts salient information from the dialogue and subsequently converts it into the PersonaTree for downstream operations. This design is particularly suitable for weaker or less capable models.

* **`pipline.py`**: Implements the main pipeline that directly converts dialogue history into operations over the PersonaTree. This is the current experimental setting used in our evaluations.



## **✨ Core Contributions**
1. **The PersonaTree Architecture:** Departing from standard vector databases or static profiles, the authors propose **PersonaTree**, a hierarchical user profiling system grounded in the **Biopsychosocial** model. This structure allows for the dynamic compression of user data, maintaining a high signal-to-noise ratio by organizing implicit user cues into a manageable tree structure with defined branches and leaves.
2. **The MemListener Model & RL Training Strategy:** To automate memory evolution, the authors introduce **MemListener**, a lightweight model trained via reinforcement learning with process-based rewards. This model is trained to interpret unstructured dialogue streams and execute precise, structured atomic operations to update the PersonaTree in real-time.
3. **Adaptive Inference & Collaborative Paradigm:** The work proposes an adaptive response mechanism that balances efficiency and depth. It employs a fast mode (reasoning directly based on the PersonaTree) for latency-sensitive scenarios and an agentic recall mode for complex queries. This establishes a new operational paradigm: "small models maintain memory while LLMs handle generation", ensuring consistent persona presentation without the computational bloat of full-context concatenation.
