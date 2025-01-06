---
tags:
  - transformer
  - 机器学习
---
![[Pasted image 20250106111814.png]]
**训练的时候，最下面的输入输出都是需要给数据的。这两部分已经有嵌入矩阵，参数需要通过训练来调整。这部分就相当于准备好词典，把单个token的词义都查出来，而理解词和词组合后的语义，靠的就是==注意力机制==（图中橙色部分）。

**词嵌入已经解决了单个token语义的问题，注意力机制要解决的是许多个词组合起来后，整体表达的语义
![[Pasted image 20250106122055.png]]
	除以根号Dout是为了化为标准正态

注意力看到后面有点听不懂了，后面再看吧
【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】 https://www.bilibili.com/video/BV1XH4y1T76e/?share_source=copy_web&vd_source=be2647d81372bbbd9ec6c17449d6b101