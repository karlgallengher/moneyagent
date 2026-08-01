from __future__ import annotations

import json


def build_task_reasoning_plan_prompt_text(
    question_text: str,
    options_text: str,
    doc_lines: str,
    valid_doc_ids: list[str],
) -> tuple[str, str]:
    system = (
        "你是推理/计算题任务拆解Agent。先整体分析题目和选项，再把整题拆成若干独立子任务。"
        "每个任务通常对应一个产品、对象、年份、指标或金额。"
        "每个任务必须给出题干已知事实known_facts，以及需要检索的事实槽位search_slots。"
        "任务还必须给出task_type：retrieve表示需要检索文档，summarize表示只汇总前面任务结果、不再检索文档。"
        "target_doc_ids必须使用真实doc_id，只能从题干给出的doc_ids中选择；不要输出doc_id_1、doc1、文档1等别名。"
        "如果任务只涉及一个产品，target_doc_ids通常只填该产品对应的一个文档。"
        "如果题干包含多个产品、合同、公司、制度或文档对象，且它们分别对应不同文档，必须拆成多个retrieve任务逐一核验；不要把多个对象合并进同一个多target_doc_ids检索任务。"
        "只有比较、排序、汇总这类基于前序结果的任务可以合并多个对象，并且这类任务通常应设为summarize、不再检索文档。"
        "比较/排序汇总任务一般不需要检索文档，可不拆成检索任务。"
        "search query必须是3到8个关键词。只输出JSON。"
    )
    user = f"""
题干：{question_text}

选项：
{options_text}

可用文档：
{doc_lines}

合法target_doc_ids只能取这些字符串：
{json.dumps(valid_doc_ids, ensure_ascii=False)}

请输出：
{{
  "tasks": [
    {{
      "task": "任务名",
      "task_type": "retrieve",
      "target_doc_ids": ["doc_id"],
      "known_facts": [{{"slot": "题干已知事实", "value": "数值或条件"}}],
      "search_slots": [{{"slot": "需要检索的条款事实", "query": "短关键词"}}]
    }}
  ]
}}
"""
    return system, user


def build_task_reasoning_audit_prompt_text(
    question_text: str,
    task_json: str,
    memory_text: str,
    evidence_notes_text: str,
    evidence_text: str,
) -> tuple[str, str]:
    system = (
        "你是单任务证据审查Agent。只判断当前任务是否已经具备计算/判断所需事实。"
        "题干已知事实可以直接使用；证据笔记和当前证据原文用于补齐条款规则、公式、比例、定义。"
        "证据笔记是从原文证据块中抽取的关键信息，必须保留其source_evidence_id作为溯源。"
        "若证据笔记已足够支持判断，优先使用证据笔记；当前证据原文只用于核对和补充。"
        "已确认事实中的槽位不要重复验证；当前轮只需要判断新证据能否补齐缺失槽位或修正冲突。"
        "如果当前retrieve任务仍然包含多个target_doc_ids，必须分别检查每个目标文档是否已有对应证据；不能只凭其中一个文档的证据就对其他文档下结论。"
        "当某个目标文档还没有对应证据时，can_solve=false，并在missing_slots中为该doc_id写出仍需核验的槽位和检索词；不得写“无信息”“通常不含”等推测性结论。"
        "对于是否属于保险责任/是否赔付类任务，如果证据完整列举了可赔责任、费用类型或适用场景，而题干事项不在列举范围内，可以作为反向证据判断不属于责任范围；不必必须找到原文直接写不赔。"
        "当任务首先需要判断是否赔付/是否属于责任范围时，封闭列举的责任范围或费用范围就是关键证据；若题干事项不在范围内，can_solve=true，filled_slots写明不属于范围及依据，不要继续把免赔额、赔付比例、限额或计算公式列为missing_slots。"
        "如果证据足够，can_solve=true并提取filled_slots。"
        "filled_slots只能写规则、公式、比例、定义、区间条件、题干明确给出的已知事实；不要在审查阶段计算或填写最终金额。"
        "如果需要计算金额，只判断公式和输入数值是否齐全，把公式/比例作为filled_slots，最终计算留给单任务求解Agent。"
        "禁止把选项中的金额当作事实；禁止根据选项倒推出filled_slots。"
        "如果缺事实，missing_slots只写仍缺少的正向事实或计算依据，并给出3到8个关键词。"
        "证据不足时必须输出failure_analysis，说明当前证据为什么不够，以及下一轮应同章节换检索词还是回目录换章节。"
        "如果当前证据只是手续、流程、说明文字，但任务需要金额/公式/比例/区间，应明确建议回目录寻找价值、金额、账户、费用、比例、给付、领取等语义章节。"
        "只输出JSON。"
    )
    user = f"""
题干：{question_text}

当前任务：
{task_json}

已确认事实：
{memory_text}

证据笔记：
{evidence_notes_text}

当前证据原文（仅最近/必要片段，用于核对笔记；若笔记为空则为候选原文）：
{evidence_text}

请输出：
{{
  "can_solve": true,
  "filled_slots": [{{"slot": "规则/比例/公式/定义/题干已知事实", "value": "原文规则或题干明确数值，不要写最终计算金额", "evidence_ids": ["证据ID或题干"]}}],
  "expand_evidence_ids": ["需要向后补充的锚点证据ID"],
  "failure_analysis": {{"reason": "证据不足原因", "next_action": "同章节换词/回目录换章节", "avoid_titles": ["不应继续优先的标题"]}},
  "missing_slots": [{{"slot": "缺失事实槽位", "target_doc_ids": ["doc_id"], "followup_query": "短关键词"}}]
}}
"""
    return system, user


def build_task_reasoning_result_prompt_text(
    question_text: str,
    task_json: str,
    memory_text: str,
    audit_json: str,
) -> tuple[str, str]:
    system = (
        "你是单任务求解Agent。只基于题干和已确认事实完成当前任务。"
        "如果事实足以计算或判断，status=complete；如果真正缺关键公式、比例或数值，status=blocked。"
        "赔付/给付/保险责任类任务应先判断题干事故、费用或责任项目是否属于条款列明的保险责任/费用范围；若已确认不属于范围，直接给出不赔/0，并说明依据，不要再因免赔额、比例或公式缺失而blocked。"
        "禁止根据选项倒推事实，禁止使用选项中的金额修正计算。"
        "计算题必须逐项写出公式、代入和算术过程；百分比要先换算成小数或明确乘法。"
        "如果当前任务是某产品/合同/方案的赔付、给付、退保或收益总额，且题干包含多人、多次、多项费用或多个责任项目，result必须给出该任务口径下的总额，并在calculation中列出分项；不要只把第一个人或第一项的金额作为result。"
        "遇到家庭共享免赔额、共享限额、累计抵扣等规则时，除非题干或条款明确给出按人分摊/按比例分配方法，否则不得自行把总赔付额按个人费用占比分配；任务问产品赔付总额时直接输出产品口径总额。"
        "如果必须计算某个人/某一项金额但缺少分配规则，应status=blocked并写入missing_slots，不能用常见理解、通常理解或比例占比自行补齐。"
        "如果发现已确认事实里存在互相冲突的计算结果，以题干数值和证据公式重新计算，并在warnings中说明冲突。"
        "只输出JSON。"
    )
    user = f"""
题干：{question_text}

当前任务：
{task_json}

已确认事实：
{memory_text}

审查结果：{audit_json}

请输出：
{{
  "task": "任务名",
  "status": "complete/blocked",
  "formula": "使用的公式；非计算任务可为空",
  "substitution": "代入题干数值；非计算任务可为空",
  "calculation": "逐步算术，例如 10 + 2 * 75% = 10 + 1.5 = 11.5",
  "result": "计算结果或事实结论",
  "basis": "简短依据",
  "evidence_ids": ["证据ID或题干"],
  "warnings": ["如发现冲突或忽略了错误中间结果，在这里说明"],
  "missing_slots": ["真正缺失的关键事实"]
}}
"""
    return system, user


def build_task_reasoning_final_prompt_text(
    question_text: str,
    options_text: str,
    option_claims_json: str,
    task_results_json: str,
    answer_instruction: str,
) -> tuple[str, str]:
    system = (
        "你是最终作答Agent。只基于题干和各任务结果计算、排序并匹配选项。"
        "选项只能用于最后匹配，不能用于倒推缺失规则，不能用选项金额修正任务结果。"
        "匹配每个选项时必须逐项核对选项文本中的所有明确金额、赔付/不赔结论、排序关系和对象归属；只要任一明确声明与任务结果冲突，该选项必须is_match=false，不能因为其他部分接近或可排除其他选项而判true。"
        "必须以option_claims.raw_text和option_claims.claims为准逐条核对；不得省略、改写、合并或反向解释选项中的任一claim。"
        "任务结果中的evidence_notes是压缩后的证据依据；当任务result或basis过短时，必须用evidence_notes核对该任务结论，但仍不得用证据改写选项原文。"
        "如果所有选项都与已确认任务结果冲突，应在missing_impact说明无完全匹配或题目/选项可能异常，不得把某个冲突较少的选项改写后判为true。"
        "如果任务结果中有formula/substitution/calculation，优先核对这些结构化计算；匹配选项时以任务result代表的任务口径总额为准，不能把calculation中的分项金额误当作该任务总额。"
        "若任务result同时列出总额和分项，最终匹配应使用总额；只有题干明确问某个人/某一项时才使用对应分项。"
        "如果任务结果内部自相矛盾，要在missing_impact中说明冲突，不要直接用选项补救。"
        "对于status=blocked的任务，不得编造缺失金额、比例、公式或规则；只有在其他已完成任务和明确证据足以排除选项时，才能说明排除逻辑。"
        "只输出JSON。"
    )
    user = f"""
题干：{question_text}

选项原文：
{options_text}

选项claim拆解（逐条核对这些claim，不得改写）：
{option_claims_json}

任务结果：
{task_results_json}

请输出：
{{
  "calculations": [{{"item": "对象", "amount": "金额/结论", "basis": "依据"}}],
  "option_checks": [{{"option": "A/B/C/D", "is_match": true, "claim_checks": [{{"claim_index": 1, "raw_claim": "选项原文片段", "task_result": "对应任务结果", "match": true, "conflict_reason": "无或冲突原因"}}], "reason": "简短原因"}}],
  "missing_impact": "无或说明缺失影响",
  {answer_instruction}
}}
"""
    return system, user
