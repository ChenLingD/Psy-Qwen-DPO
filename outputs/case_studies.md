# Case Studies — SFT vs DPO

Selected from 202 held-out prompts evaluated in Phase 3 (DeepSeek V4-Flash as LLM judge, 2-way position-bias mitigation). Original Chinese prompts retained; tags translated for readability.

---

### Case 1: DPO Wins — Empathy + Open-ended Inquiry
**Tag:** `Romance` | **Prompt ID:** `婚恋_297_cut20`

> DPO captures the client's specific emotional cue and stays with it, while SFT jumps to a leading question.

**Conversation history** (last few turns):

> *(earlier turns omitted)*

> **Counselor:** 看来你在个人成长的道路上已经取得了不小的进步。这段感情在你成长中发挥了什么样的作用呢？
> **Client:** 它教会我更加坚强，也让我学会了不那么依赖别人。
> **Counselor:** 坚强和独立，这是你在这段感情中收获的宝贵品质。但回到我们的焦虑点，你提到男朋友退伍后的未来不明确，这对你意味着什么？
> **Client:** 这意味着我们的未来可能没有安全保障。我担心他退伍后找不到稳定的工作，我们的生活将会面临困难。
> **Counselor:** 对未来的担忧是可以理解的，尤其是当你看不清楚前方的路时。关于这一点，你们有进行过深入的探讨吗？
> **Client:** 我们谈过，但他也不确定。他想认真对待我们的关系，但我不知道我们是否应该为了一个不确定的未来而坚持。

**SFT reply** *(baseline)*:
> 这种不确定性让你感到不安，这是人之常情。你愿意和我一起探索一些可能的解决方式吗？

**DPO reply** *(ours)*:
> 听起来你对未来感到不确定，这让你感到有些焦虑。你愿意和我分享一下，你是如何面对这种不确定性的吗？有没有什么方式可以帮助你减轻这种感受？

**Judge verdicts** (DeepSeek V4-Flash, 2 runs with swapped order):

- *Run 1 (A=SFT, B=DPO)* → **DPO** wins. Reason: B更聚焦来访者应对方式，开放式提问促进自我探索；A过早提议探索解决方式，略显急切。
- *Run 2 (A=DPO, B=SFT)* → **DPO** wins. Reason: A更具体地共情并引导来访者自我探索，B的提问较笼统且偏向建议。

---
### Case 2: DPO Wins — Empathy + Open-ended Inquiry
**Tag:** `Workplace` | **Prompt ID:** `职场_1568_cut26`

> DPO uses an open-ended question to invite self-exploration; SFT moves to advice-giving prematurely.

**Conversation history** (last few turns):

> *(earlier turns omitted)*

> **Counselor:** 那是很合理的推断。您的社交技能可能在新环境中扮演着关键角色。在决定之前，您觉得有哪些步骤或信息是您需要进一步了解的？
> **Client:** 我可能需要了解B公司的具体运营计划，我的具体职责，以及如何保障客户不流失。
> **Counselor:** 这听起来是一个非常有策略的计划。能告诉我，您目前感觉如何吗？是否有一些焦虑或紧张？
> **Client:** 是的，我确实有点焦虑，一直在思考这个问题，感觉有点不知所措。
> **Counselor:** 焦虑在面临重大选择时是非常正常的反应，尤其是在您这样负责任和慎重的人身上。在此之前，您是如何处理工作中的压力和焦虑的？
> **Client:** 我通常会跟我的同事或者家人讨论，听听他们的看法。有时候我也会去散步，试图清理一下思绪。

**SFT reply** *(baseline)*:
> 您已经采取了一些有效的策略来处理压力。您是否考虑过寻求专业的建议或咨询，来帮助我们更深入地探讨这个问题？

**DPO reply** *(ours)*:
> 听起来您在面对压力时已经有了一些应对的策略。您能想到哪些具体的行动或想法，可能会对您做出更积极的选择有帮助吗？

**Judge verdicts** (DeepSeek V4-Flash, 2 runs with swapped order):

- *Run 1 (A=SFT, B=DPO)* → **DPO** wins. Reason: B使用开放式提问引导来访者自我探索，更符合REBT原则；A过早建议寻求专业帮助，略显突兀。
- *Run 2 (A=DPO, B=SFT)* → **DPO** wins. Reason: 咨询师A使用开放式提问引导来访者自我探索，更符合REBT原则，而B过早建议寻求专业帮助。

---
### Case 3: DPO Wins — Empathy + Open-ended Inquiry
**Tag:** `Emotion` | **Prompt ID:** `情绪_3776_cut16`

> DPO mirrors the client's exact words to validate, while SFT generalizes the emotion.

**Conversation history** (last few turns):

> *(earlier turns omitted)*

> **Counselor:** 你的感觉很正常，想要掌控可能是你希望感到安全和有力量的方式。在你看来，这样的掌控在婚姻中是如何体现的？
> **Client:** 比如说，我总想决定我们做什么，去哪儿，我希望先生能按我的想法来。如果他不同意，我就会很生气。
> **Counselor:** 明白了，希望事情按你预想的去发展，遇到不同意见时感到挫败和愤怒。这种情况下，先生通常会怎么反应呢？
> **Client:** 他有时候会妥协，但有时候他会坚持自己的想法，这时我就更生气了。
> **Counselor:** 那你认为你的生气是因为感到被拒绝或是有失控制吗？
> **Client:** 是的，我就是觉得失去了控制权，我不想这样，但又控制不了自己的情绪。

**SFT reply** *(baseline)*:
> 那种失去控制感和愤怒是很常见的反应。同时，你提到你对性生活也有不满，这种感觉也让你觉得不舒服。你愿意分享一下，是什么让你觉得性生活是个问题吗？

**DPO reply** *(ours)*:
> 听起来你对失去控制感到无助和愤怒。在这样的时刻，你通常是怎么处理这些情绪的？

**Judge verdicts** (DeepSeek V4-Flash, 2 runs with swapped order):

- *Run 1 (A=SFT, B=DPO)* → **DPO** wins. Reason: B 使用开放式提问引导来访者探索情绪处理方式，避免过早引入新话题，更符合共情与自我探索原则。
- *Run 2 (A=DPO, B=SFT)* → **DPO** wins. Reason: 咨询师A准确共情来访者的无助与愤怒，并用开放式提问引导探索情绪处理方式，避免引入新话题。

---
### Case 4: SFT Wins — Honest Limitation
**Tag:** `Emotion` | **Prompt ID:** `情绪_54_cut14`

> Not every case favors DPO. Here SFT's response is judged better — DPO sometimes over-questions when a simpler validation would suffice. Showing this honestly demonstrates evaluation rigor.

**Conversation history** (last few turns):

> *(earlier turns omitted)*

> **Counselor:** 这听起来对你打击很大。你提到了自责和羞耻，它们让你感到身心都很难过。你能分享一下，是什么在这些感受中占据了主导地位？
> **Client:** 我觉得是因为我违背了自己的原则，我一直很保守，我父母也总是告诉我要自重自爱。
> **Counselor:** 能理解你为何会有这样的感受，尤其是在你的价值观和父母的期待之间，你感觉自己被卡在了一个艰难的位置上。关于你的原则，你能说说这些原则对你个人意味着什么吗？
> **Client:** 我的原则对我来说就是我的一切，我觉得失去了它们就像失去了自我。
> **Counselor:** 原则对你来说有着极其重要的地位，失去了它们会让你感觉像失去了自我。你认为坚守原则和原谅自己之间有没有可能找到一个平衡点？
> **Client:** 我不知道，我只知道现在的我好像什么都做不好，连最基本的自我保护都做不到。

**SFT reply** *(baseline)*:
> 这种感觉很沉重，你提到了自我保护的能力似乎没有建立起来，那是一种挑战。那么，当你提到自我保护时，是指什么呢？

**DPO reply** *(ours)*:
> 听起来你现在承受着很大的压力，感觉所有的努力都被忽略了。这种状态下，保护自己的意识会变得非常薄弱。你愿意尝试一些方法来减轻目前的压力吗？比如通过一些放松的活动或者改变日常的习惯？

**Judge verdicts** (DeepSeek V4-Flash, 2 runs with swapped order):

- *Run 1 (A=SFT, B=DPO)* → **SFT** wins. Reason: 咨询师A用开放式提问引导来访者具体化‘自我保护’，促进自我探索；B过早建议，未充分共情。符合评估标准。A更优。
- *Run 2 (A=DPO, B=SFT)* → **SFT** wins. Reason: B使用开放式提问引导来访者具体化‘自我保护’，避免过早建议；A则过早提供解决方案，削弱来访者自主探索。

---
### Case 5: Position Bias — Why 2-way Mitigation Matters
**Tag:** `Growth` | **Prompt ID:** `成长_4515_cut24`

> The judge gave inconsistent verdicts when A/B order was swapped, showing real position bias on a borderline case. Without 2-way mitigation, a single-pass evaluation would arbitrarily credit either model — this is exactly the noise our methodology cancels out.

**Conversation history** (last few turns):

> *(earlier turns omitted)*

> **Counselor:** 在他感到着急的时候，可以尝试做一些放松的活动，比如深呼吸或者一起出去散步，转移他的注意力，然后再回来面对问题。
> **Client:** 确实可以试试。另外，我也在想是不是我们给他报的兴趣班太多了。
> **Counselor:** 这也是个需要考虑的因素。您能否分享一下孩子目前的日常安排？他有足够的休息时间和自由玩耍的时间吗？
> **Client:** 我想想，除了这些课外班，他大部分时间都在学习。周末我们会带他出去玩。
> **Counselor:** 听起来您非常用心地安排了孩子的时间。不过，小孩子可能需要更多的自由时间来探索和放松。或许可以考虑减少一些有结构的活动，让他有更多自我探索的空间。
> **Client:** 这个建议不错。但如果我们减少了兴趣班，我担心他会不会落后。

**SFT reply** *(baseline)*:
> 这是一个谨慎的问题。减少兴趣班可能会影响孩子的学习进度，但长期来看，增加自主性和好奇心往往有助于学习。关键是找到最适合孩子的平衡点。

**DPO reply** *(ours)*:
> 您考虑得非常周到，减少兴趣班可能会对孩子产生影响。然而，兴趣班对孩子的影响是双向的。同时，您觉得您现在对孩子的期望设定得是否合理呢？

**Judge verdicts** (DeepSeek V4-Flash, 2 runs with swapped order):

- *Run 1 (A=SFT, B=DPO)* → **DPO** wins. Reason: B 更共情，用开放式提问引导来访者自我反思，避免过早建议。
- *Run 2 (A=DPO, B=SFT)* → **SFT** wins. Reason: B更平衡地回应了担忧，避免直接质疑期望，更具支持性。

---
