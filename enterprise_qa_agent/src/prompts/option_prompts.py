from __future__ import annotations

import json


def build_doc_router_prompt_text(
    question_text: str,
    option_label: str,
    option_claim: str,
    doc_lines: str,
    profile_lines: str,
    hint_docs: list[str],
) -> tuple[str, str]:
    system = (
        "你是Doc Router。判断当前选项需要检索哪些参考文档。只输出JSON。"
        "如果当前选项出现明确产品名、机构名、法规名，应优先选择文档摘要中直接包含该名称或高度相近名称的文档。"
    )
    user = f"""
题干：{question_text}
当前选项：{option_label}. {option_claim}

可用文档：
{doc_lines}

文档摘要：
{profile_lines}

程序预匹配的候选文档（若非空，通常应优先考虑）：
{json.dumps(hint_docs, ensure_ascii=False)}

请输出：
{{
  "target_doc_ids": ["doc_id"],
  "doc_reasons": {{"doc_id": "为什么需要该文档"}},
  "reason": "简短说明"
}}
"""
    return system, user


def build_slot_planner_prompt_text(
    question_text: str,
    answer_format: str,
    option_label: str,
    option_claim: str,
    entities: list[str],
) -> tuple[str, str]:
    system = (
        "你是选项槽位规划Agent。只根据题干和当前选项，拆出验证该选项所需的最小事实槽位，"
        "并为每个槽位生成用于全文检索的短关键词。不要选择文档，不要判断选项对错。只输出JSON。"
    )
    user = f"""
题干：{question_text}
题型：{answer_format}
当前选项：{option_label}. {option_claim}
题目实体：{json.dumps(entities, ensure_ascii=False)}

请输出：
{{
  "slots": [
    {{
      "slot": "需要验证的事实",
      "expected": "选项声称的值/关系，可为空",
      "queries": ["语义检索短关键词", "数值/时间检索短关键词"],
      "key_terms": ["当前槽位必须覆盖的对象/指标/条件/关系关键词"]
    }}
  ]
}}
要求：
1. 只拆当前选项本身需要验证的事实，不要加入无关背景。
2. query 应包含对象、时间、指标；如果选项含数值或方向，至少一个 query 包含该数值或方向。
3. 必须保留选项中的关键口径词，不要改成相近但不同的指标；例如“研发投入占营业收入比例”不要改成“研发费用 营业收入 比例”，“每股”不要改成“每10股”，“现金分红金额占归母净利润”不要改成“本次利润分配中占比”。
4. query 不要写成长句，使用空格分隔关键词。
5. key_terms 来自当前题干和选项，不要写行业通用词表；每个槽位3-8个。
6. 一般输出1-2个slots，每个slot最多2个queries。
7. 如果当前选项是“都/均/全部/所有/每个”等全称断言，且题目实体包含多个对象，必须为每个对象分别输出槽位；判断支持时需要全部对象都有证据，判断不支持时可由任一明确反例支持。
8. 如果当前选项包含“因为/由于/原因是”等因果解释，必须把结论和每个原因分别拆成槽位；“且/并且/以及/同时”连接的多个原因不能合并为一个槽位。
"""
    return system, user


def build_evidence_read_prompt_text(
    question_text: str,
    option_label: str,
    option_claim: str,
    slots_json: str,
    memory_text: str,
    evidence_text: str,
) -> tuple[str, str]:
    system = (
        "你是证据阅读Agent。你的任务只是在候选证据中抽取与待验证槽位有关的事实。"
        "不要判断选项最终对错，不要决定是否丢弃证据，不要生成下一轮检索词。"
        "如果一个证据只能支持某个槽位的一部分，也要抽取出来，并在coverage写partial。"
        "事实必须来自证据原文、题干已知事实或基于证据数字的明确计算；禁止照抄选项声称当事实。"
        "只输出JSON。"
    )
    user = f"""
题干：{question_text}
当前选项：{option_label}. {option_claim}

待验证槽位：
{slots_json}

已有压缩事实：
{memory_text}

本轮/近期证据：
{evidence_text}

请输出：
{{
  "facts": [
    {{
      "slot": "该事实对应的槽位或子槽位",
      "value": "从证据中抽取的事实、数值、关系、公式或边界条件，最多180字",
      "coverage": "full|partial",
      "evidence_ids": ["证据ID"]
    }}
  ]
}}
要求：
1. 每条事实必须引用至少一个证据ID。
2. 能证明部分对象、部分指标、分子/分母之一、关系的一侧，也要输出coverage=partial。
3. 如果证据只含泛泛背景且完全不能支持任何槽位，facts输出空数组。
4. 最多输出6条facts，优先覆盖不同槽位和不同目标对象；不要重复输出已有压缩事实。
"""
    return system, user


def build_audit_prompt_text(
    question_text: str,
    option_label: str,
    option_claim: str,
    target_doc_ids: list[str],
    slots_json: str,
    memory_text: str,
    evidence_facts_text: str,
    evidence_text: str,
) -> tuple[str, str]:
    system = (
        "你是槽位状态审查Agent。判断已抽取事实和当前证据是否足以验证当前选项；不足则给出缺失槽位和短检索词。"
        "优先使用已抽取事实；当前证据正文只用于核对和补充。"
        "单位、期间、分母、统计口径必须一致或明确可换算；“每10股”和“每股”、“研发费用”和“研发投入”、“年度现金分红”和“年度+特别现金分红”不能直接视为同一口径。"
        "如果原文同时给出组成项和合计项，必须按当前选项措辞对应的最小口径抽取；选项未写“合计/总额/含其他方式/特别”等词时，不要自行扩大为合计口径。"
        "例如“同比下降18.97%”可以支持“增速=-18.97%”，“同比增长14%”可以支持“增速=14%”。"
        "filled_slots中的value必须来自证据原文、题干已知事实或基于证据中数字的明确计算；禁止把当前选项声称的数值直接当作已确认事实。"
        "你必须先在内部明确验证当前选项所需的最小事实集合；只有这些事实都能由题干、已确认事实或当前证据支持时，can_judge才能为true。"
        "只要任一必要事实缺失，就必须把该事实写入missing_slots，并设置can_judge=false。"
        "filled_slots禁止写半截结论或待计算结论；value中不得出现问号、未知、未直接给出、需计算、可反推、已比较等不完整表述。"
        "比较/比例类选项必须拿到双方对象的可比较数值或可直接支持的原文比例，缺任一方都必须进入missing_slots。"
        "因果解释类选项必须同时核对结论和每个原因；任一原因没有直接证据，或证据把该原因归属于其他对象/地区/期间/口径，都不能视为已填充。"
        "含“主要、集中、重点、核心、首要”等程度词的槽位，证据必须支持该程度；若原文是多个并列对象或多个核心区域，不能支持“主要集中在其中一个对象”。"
        "证据中的限定对象必须和槽位对象一致；例如某因素属于A地区，不能迁移为B地区的原因。"
        "禁止用常识、估算、行业经验、选项倾向、排除法或未经证据支持的假设来补足缺失事实。"
        "不要判断证据块是否永久无关，不要输出需要丢弃的证据ID。"
        "只输出JSON，且只能使用can_judge、filled_slots、missing_slots三个顶层字段。"
    )
    user = f"""
题干：{question_text}
当前选项：{option_label}. {option_claim}
目标文档：{json.dumps(target_doc_ids, ensure_ascii=False)}

待验证槽位：
{slots_json}

已确认事实：
{memory_text}

本轮从证据中抽取的事实：
{evidence_facts_text}

当前证据：
{evidence_text}

请输出：
{{
  "can_judge": true,
  "filled_slots": [{{"slot": "事实槽位", "value": "事实值", "evidence_ids": ["证据ID"]}}],
  "missing_slots": [{{"slot": "缺失槽位", "target_doc_ids": ["doc_id"], "followup_query": "短关键词", "key_terms": ["下一轮证据必须覆盖的关键词"]}}]
}}
不要输出matching_slots、matched_slots、matched_evidence等其他字段。若证据表述为“同比下降18.97%”，可将事实值写为“-18.97%（同比下降18.97%）”。
如果当前选项需要比较、计算或判断多个对象，请分别检查每个对象的必要事实是否都有证据。缺少任何一项时，不要进入最终判断。
如果当前选项需要计算占比、比例、比重、强度或率，分子和分母可以来自不同证据块；只要对象、年份、单位和口径一致，就应合并计算并填入filled_slots。
如果只拿到了分子但没有分母，或只拿到一方比例但另一方比例未知，必须把缺少的分母/比例写入missing_slots，不得把“高于/低于/已比较”写入filled_slots。
不要因为槽位名称和原文措辞不完全一致就判为缺失；如果原文能够等价支持该事实，应填入filled_slots并引用证据ID。
如果选项中的数值没有出现在证据中，且不能由证据中的原始数字直接计算得到，必须判为缺失或填入证据中的实际值，不得照抄选项数值。
如果证据只支持相近但不同口径的事实，应把正确口径写入filled_slots；若还缺少当前选项口径，则写入missing_slots继续检索。
如果同一证据同时列出单项、特别项、其他方式、合计/总额等多种口径，应分别识别；当前选项没有明确要求合计时，优先抽取与选项文字最贴近的单项或原文同名项目。
如果选项用“因为/由于/且/并且”给出多个原因，必须逐一填充每个原因；原因A有证据不能替代原因B。若原文把某原因写给其他对象、地区或市场，应把当前对象下该原因写入missing_slots。
如果选项声称“主要集中/重点集中/主要目的地”，但证据只说“两个核心区域/多个重点/其中之一”，不得填为“是”，应写入missing_slots或给出实际关系。
missing_slots 的 key_terms 必须来自缺失槽位和当前题目，不要写行业通用词表；用于后续证据排序。
"""
    return system, user


def build_judge_prompt_text(
    question_text: str,
    answer_format: str,
    option_label: str,
    option_claim: str,
    memory_text: str,
    evidence_facts_text: str,
    audit_json: str,
    evidence_text: str,
) -> tuple[str, str]:
    system = (
        "你是做题Agent。只能基于给定证据判断当前选项陈述本身是否正确。"
        "verdict=true表示当前选项陈述正确，verdict=false表示当前选项陈述错误或证据不足。只输出JSON。"
        "已确认事实是审查Agent从证据中抽取的结构化线索；证据原文和基于同口径数字的可复算结果具有更高优先级。"
        "如果已确认事实与证据原文、表格数字或你重新计算出的关系冲突，必须以证据原文和重新计算结果为准，并在reason中说明冲突。"
        "如果已确认事实中已经包含某对象、指标、年份或数值，不要因为同一证据块还包含其他对象的信息而忽略或否定该事实。"
        "如果当前选项包含高于、低于、多于、少于、大于、小于、快于、慢于、均为、是否等关系，必须先识别选项声称的关系，再用证据计算或抽取实际关系，最后检查二者是否一致。"
        "如果证据显示的关系与选项声称的方向相反，verdict必须为false。"
        "必须逐字核对单位和统计口径；每股、每10股、每手、每百元、含税/不含税、年度现金分红、特别现金分红、合计现金分红、研发费用、研发投入、占营业收入比例等口径不一致时，除非证据明确给出换算并换算后相同，否则verdict=false。"
        "如果证据同时存在组成项和合计项，应先按当前选项文字选择最贴近的同名/同义项目；选项未明确写合计、总额、含其他方式、年度+特别等词时，不要主动改用合计项否定它。"
        "如果已确认事实来自相近但不同口径的证据，不能据此判当前选项为true。"
        "当题干和当前选项已经给出明确比较对象和指标时，不要改用其他年份、其他口径或额外背景事实推翻当前证据支持的同口径判断。"
        "当前选项若包含多个原因、并列条件或因果解释，必须逐项核对；任一原因无直接证据或归属对象不一致，verdict不能为true。"
        "当前选项若包含主要、集中、重点、核心等程度词，证据必须支持该程度；证据只说多个并列对象、多个核心区域或其中之一时，不能支持“主要集中在某一个对象”。"
        "如果审查结果仍有missing_slots，说明当前必要事实未齐全；不得用题干背景、没有反例、行业常识或猜测把缺失事实补成true。"
        "题干背景只能提供待核验范围，不能替代证据证明选项中的新增关系、数值或结论。"
    )
    user = f"""
题干：{question_text}
题型：{answer_format}
当前选项：{option_label}. {option_claim}

已确认事实：
{memory_text}

证据阅读事实：
{evidence_facts_text}

审查结果：
{audit_json}

证据：
{evidence_text}

请输出：
{{
  "verdict": true,
  "confidence": 0.0,
  "reason": "简短依据",
  "answer": "若当前选项正确则输出当前选项字母，否则输出空字符串；判断题可输出A或B"
}}
注意：不要把“已经完成判断”理解为verdict=true；verdict只表示当前选项陈述是否正确。verdict、answer、reason必须三者一致：reason支持选项成立时verdict=true；reason认为选项错误、不一致或证据不足时verdict=false。若理由中认为当前选项错误，verdict必须为false。
reason必须说明：选项声称的关系是什么、证据得到的关系是什么、二者是否一致。比较方向相反时不要输出当前选项字母。
reason还必须说明单位和口径是否一致；例如证据为“每10股派息45.53元”而选项为“每股45.53元”时，应判为false。
判断时可以先使用“已确认事实”中的对象-指标-数值作为线索，但遇到比较/比例/增长下降/是否高低这类关系时，必须用证据中的原始数值或明确同比值复核一遍。
如果一个选项可由证据中的同名组成项直接支持，不要因为同一段证据还存在更大的合计项就判为错误；只有选项明确要求合计口径时才用合计项。
如果审查结果中missing_slots非空，除非证据部分已经逐项补齐这些missing_slots并在reason中引用证据ID，否则verdict必须为false。
“题干已经提到该领域/对象”“没有找到反例”“可能存在类似案例”都不能作为verdict=true的理由。
如果选项说“因为A且B”，reason必须分别说明A和B的证据；不能用A的证据替代B，也不能把属于其他对象/地区/期间的原因迁移到当前对象。
如果证据原文是“两个核心区域/多个重点/一是X、二是Y”，而选项说“主要集中在X”，应判为false或证据不足，除非证据另有明确主次排序。
"""
    return system, user


def build_toc_repair_prompt_text(
    question_text: str,
    option_label: str,
    option_claim: str,
    missing_slots_json: str,
    recent_json: str,
    tried_sections_json: str,
    toc_text: str,
) -> tuple[str, str]:
    system = (
        "你是目录补证Agent。当前普通检索证据不足，你需要根据题干、当前选项、缺失槽位和文档目录，"
        "选择最可能包含答案的章节、图或表。只输出JSON。"
        "可以从全部候选文档中补选文档，不要局限于之前router选中的文档。"
        "每个缺失事实优先选择最具体的图表标题或章节标题。"
    )
    user = f"""
题干：{question_text}
当前选项：{option_label}. {option_claim}

缺失槽位：
{missing_slots_json}

已有证据摘要：
{recent_json}

已尝试但仍未补齐缺失槽位的章节：
{tried_sections_json}

候选文档目录：
{toc_text}

请输出：
{{
  "section_queries": [
    {{"doc_id": "doc_id", "section_hint": "目录中的章节/图/表标题", "query": "进入该章节后用于核验证据的短关键词"}}
  ],
  "reason": "为什么这些章节能补齐缺失槽位"
}}
最多输出3个section_queries。doc_id必须来自候选文档。section_hint应尽量原样复制目录标题。
如果已尝试章节只命中相近但不直接回答的内容，必须改选同一文档内其他更相关章节、相邻条款或上级目录下的其他子条款。
"""
    return system, user


def build_evidence_extract_prompt_text(
    question_text: str,
    option_label: str,
    option_claim: str,
    missing_slots_json: str,
    evidence_items_json: str,
) -> tuple[str, str]:
    system = (
        "你是块内证据摘录Agent。你只能从给定证据块原文中摘录能验证当前选项的句子、短段落或表格行。"
        "不要解释、不要改写、不要推理、不要补充原文没有的内容。只输出JSON。"
    )
    user = f"""
题干：{question_text}
当前选项：{option_label}. {option_claim}

上一轮缺失槽位：
{missing_slots_json}

证据块：
{evidence_items_json}

请输出：
{{
  "extractions": [
    {{"source_evidence_id": "原证据ID", "text": "从原文逐字摘录的相关句子/表格行，最多500字", "reason": "对应哪个事实槽位"}}
  ]
}}
要求：
1. text 必须是原文连续片段或表格的原始行组合，不得改写。
2. 优先摘录同时包含选项中的实体、年份、指标、数值的句子。
3. 正文中已有精确数值时，优先摘正文句子；正文没有时再摘表格行。
4. 最多输出4条；如果没有相关原文，输出空数组。
"""
    return system, user
