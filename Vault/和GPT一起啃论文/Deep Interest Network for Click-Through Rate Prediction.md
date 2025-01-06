
---

## **Abstract（摘要）**

1. **背景与问题**：
    
    - 点击率（CTR）预测在广告系统中至关重要。CTR决定了广告的排名和收入。
    - 当前主流的深度学习方法多采用 **Embedding&MLP** 框架：
        - 将稀疏输入特征映射为低维嵌入向量。
        - 将用户的兴趣压缩成固定长度的向量。
    - **问题**：
        - 固定长度的用户兴趣表示无法捕捉用户的多样化兴趣，尤其是广告与用户行为的相关性。
2. **解决方案**：
    
    - 提出 **Deep Interest Network (DIN)**：
        - 引入局部激活单元（Local Activation Unit），根据候选广告动态生成用户兴趣表示： $vU(A)=∑j=1Ha(ej,vA)ejv_U(A) = \sum_{j=1}^H a(e_j, v_A) e_j 其中：$
            - $eje_j$：用户行为的嵌入表示。
            - $vAv_A$：广告的嵌入表示。
            - $a(ej,vA)a(e_j, v_A)$：表示用户行为与广告的相关性权重。
3. **创新技术**：
    
    - **小批量感知正则化**：降低大规模参数计算开销。
    - **数据自适应激活函数**：动态调整输入分布的阈值。
4. **成果**：
    
    - 在公共数据集和阿里巴巴的真实广告数据集上表现优异。
    - 已成功部署于阿里巴巴广告系统，显著提升了CTR和收入。

---

## **1. Introduction（引言）**

1. **CTR预测的重要性**：
    
    - 在按点击计费（CPC）广告系统中，CTR预测是广告排序的关键因素。
    - CTR决定了 $eCPM=CTR×bid price\text{eCPM} = \text{CTR} \times \text{bid price}$，直接影响系统收入。
2. **现有方法的不足**：
    
    - 当前深度学习方法采用 **Embedding&MLP** 框架，将用户行为表示为固定长度向量：
        - **瓶颈**：固定长度无法表达用户多样化的兴趣。
        - **放大维度的问题**：
            - 需要增加嵌入维度以提升表达能力。
            - 导致模型参数增多，可能过拟合且增加存储和计算负担。
3. **DIN的提出**：
    
    - 基于以下观察：
        - 用户的多样化兴趣具有 **局部激活特性**（Local Activation Property）：仅与当前广告相关的行为会被激活。
    - **解决方案**：通过局部激活单元动态生成用户兴趣表示，以提高模型表达能力。

---

## **2. Related Work（相关工作）**

1. **CTR预测模型演变**：
    
    - **早期方法**：
        - Logistic Regression（LR）：依赖人工特征工程。
    - **深度学习方法**：
        - Wide&Deep、DeepFM等通过嵌入层和MLP减少特征工程需求。
        - 但用户的兴趣仍然被压缩为固定长度。
2. **注意力机制的启发**：
    
    - **来源**：神经机器翻译（Neural Machine Translation, NMT）。
    - 通过加权求和对相关信息赋予高权重。
    - DIN采用类似机制，但在CTR预测中加入广告与用户行为的动态交互。

---

## **3. Background（背景）**

1. **阿里巴巴广告系统架构**：
    
    - **匹配阶段**：筛选相关广告。
    - **排序阶段**：预测每个广告的CTR并排序。
    - 用户行为多样化：例如，年轻母亲可能浏览童装和手提包，但对广告的响应仅与部分行为相关。
2. **CTR预测的挑战**：
    
    - 用户行为包含丰富信息，但需要筛选出与广告相关的部分。

---

## **4. Deep Interest Network（DIN模型）**

### **4.1 Feature Representation（特征表示）**

- **稀疏特征**：
    
    - 数据通常为多组类别特征（如用户性别、浏览过的商品）。
    - 使用多热编码（multi-hot encoding）表示用户的行为数据。
- **公式表示**：
    
    - 对于 ii-th 特征组，其编码为：$$ ti∈RKi,ti[j]∈{0,1},∑j=1Kiti[j]=kt_i \in \mathbb{R}^{K_i}, \quad t_i[j] \in \{0, 1\}, \quad \sum_{j=1}^{K_i} t_i[j] = k$$
    - $$
        - KiK_i：特征组的维度。
        - k=1k = 1：单热编码；k>1k > 1：多热编码。$$
- 用户行为特征常通过多热编码表示，例如：
    $$
    t=[tweekday,tgender,tvisited_goods_ids,tad_cate_id]t = \left[ t_{\text{weekday}}, t_{\text{gender}}, t_{\text{visited\_goods\_ids}}, t_{\text{ad\_cate\_id}} \right]
$$
---

### **4.2 Base Model（Embedding&MLP框架）**

1. **嵌入层**：
    
    - 将稀疏的高维特征 tit_i 映射到低维嵌入向量 $$eie_i： ei=Wi⋅tie_i = W_i \cdot t_i$$
        $$- Wi∈Rd×KiW_i \in \mathbb{R}^{d \times K_i}：嵌入矩阵。
        - dd：嵌入向量维度。$$
2. **池化层**：
    
    - 对用户行为嵌入向量的列表进行池化，以生成固定长度的表示： $$ei=pooling(ei1,ei2,…,eiH)e_i = \text{pooling}(e_{i1}, e_{i2}, \dots, e_{iH})$$
        - 常用的池化方法包括 **平均池化** 和 **求和池化**。
3. **MLP层**：
    
    - 将池化后的稠密向量输入全连接网络，学习特征的非线性组合关系。
4. **问题**：
    
    - 用户兴趣表示为固定长度，不能反映广告与用户行为的动态关系。

---

### **4.3 The Structure of DIN（DIN的结构）**

1. **局部激活单元**：
    
    - 动态生成用户兴趣表示：$$ vU(A)=∑j=1Ha(ej,vA)⋅ejv_U(A) = \sum_{j=1}^H a(e_j, v_A) \cdot e_j $$其中：
        $$- eje_j：用户第 jj 个行为的嵌入向量。$$
        $$- vAv_A：候选广告的嵌入向量。$$
        $$- 激活权重 a(ej,vA)a(e_j, v_A) 由前馈神经网络建模。$$
2. **区别于传统注意力机制**：
    
    - DIN没有对权重 $a(ej,vA)a(e_j, v_A)$ 进行归一化（例如softmax），而是保留了权重的强度信息。
3. **优点**：
    
    - 用户兴趣表示会随广告动态变化。
    - 提高模型表达能力，无需显著增加计算开销。

---

## **5. Training Techniques（训练技巧）**

### **5.1 Mini-batch Aware Regularization（小批量感知正则化）**

1. **背景**：
    
    - 在工业级稀疏数据中，嵌入矩阵的参数量极大，直接计算 ℓ2\ell_2 正则化代价高昂。
2. **公式**：
    $$
    - 常规 ℓ2\ell_2 正则化： L2(W)=∑j=1K∥wj∥22L_2(W) = \sum_{j=1}^K \| w_j \|_2^2$$
    - 小批量感知正则化仅计算当前批次的非零参数：$$ L2mini-batch(W)=∑j=1Kαj⋅∥wj∥22L_2^{\text{mini-batch}}(W) = \sum_{j=1}^K \alpha_j \cdot \| w_j \|_2^2$$
        $$- αj\alpha_j：表示当前批次中第 jj 个特征是否被激活。$$

---

### **5.2 Data Adaptive Activation Function（数据自适应激活函数）**

1. **PReLU的不足**：
    
    - 固定的阈值 α\alpha 难以适应不同输入分布。
2. **Dice激活函数**：
    
    - 动态调整输入分布的阈值：$$ f(s)=p(s)⋅s+(1−p(s))⋅αs,p(s)=11+e−s−E[s]Var[s]+ϵf(s) = p(s) \cdot s + (1 - p(s)) \cdot \alpha s, \quad p(s) = \frac{1}{1 + e^{-\frac{s - E[s]}{\sqrt{\text{Var}[s] + \epsilon}}}}$$

---

## **6. Experiments（实验）**

1. **主要实验结果**：
    
    - DIN结合小批量正则化和Dice激活函数后，在阿里巴巴数据集上的AUC提升11.65%。
2. **在线A/B测试**：
    
    - CTR提升10%，收入提升3.8%。

---
