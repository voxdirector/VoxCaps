from __future__ import annotations

import re
from textwrap import dedent
from typing import List, Optional, Sequence, Tuple

from .templates import (
    VOXCAPS_AUDIO_CAPTION_TEMPLATE,
)

# 用占位符延迟注入参考文本，避免在定义模板时就和具体样本耦合。
TEXT_PROMPT_PLACEHOLDER = "__KNOWN_TEXT__"

# 把常见语言缩写/别名统一映射成 prompt 中展示给模型的中文名称。
LANGUAGE_NAME_MAP = {
    "zh": "中文",
    "zh-cn": "中文",
    "cn": "中文",
    "chinese": "中文",
    "中文": "中文",
    "en": "英文",
    "english": "英文",
    "英文": "英文",
    "ja": "日语",
    "jp": "日语",
    "japanese": "日语",
    "日文": "日语",
    "日语": "日语",
    "ko": "韩语",
    "kr": "韩语",
    "korean": "韩语",
    "韩文": "韩语",
    "韩语": "韩语",
}

ANIME_GUIDANCE_OVERVIEW = dedent(
    """
    此为动漫/动画场景，侧重刻画角色**稳定的表演风格**与**可听辨的情绪线索**。所有判断必须基于声音证据，避免空泛标签或视觉脑补。

    **1. `speakers` 的核心描述框架**
       - 描述顺序优先为：`性别` > `年龄段` > `角色原型/画像` > `稳定音色` > `常态说话风格`。
       - `speakers` 的核心是整段中反复出现、相对稳定的表演基线，例如音色、常态语速、能量感、吐字方式、口语化程度、长期情绪底色。
       - `content` 只写当前这句的即时变化，例如突然爆发、语速骤快、声音发抖、情绪翻转、气声增多、短暂失控等。
       - 对少儿群像、阵营战斗、强拟人化设定的动画，也请优先写**人格化的声音原型**，不要把 `speakers` 写成物种名、材质名或外形识别结果。

    **2. 可优先关注但不是逐项强制填写的证据维度**
       - 若证据稳定且反复出现，可补充：稳定音色、语速/能量、句尾音调走势、口语化程度、常见语气词或口头禅、具有辨识度的口部动作。
       - 这些维度是帮助你判断角色原型的“证据项”，不是每个 speaker 都必须逐项写全；只有当它们稳定、清晰、能长期支撑角色感时才写入 `speakers`。

    **3. 原型词必须落到可听见的声音证据上**
       - 不要只写“傲娇”“热血”“病娇”“三无”这类标签；应把标签落到音色、语速、音量、气息、句尾走势、语气词和情绪控制方式上。
       - 若证据不足，宁可保守描述为“带有少年感”“带有冷淡克制感”“带有元气感”，不要强行写得过细。
       - 对强拟人化动画，不要把“非人设定”直接误写成电子合成感或强处理的非自然音色；除非音频里真的有合成器、声码器或明显后期处理痕迹。
    """
).strip()

ANIME_STYLE_MATRIX_SECTION = dedent(
    """
    **4. 分众/题材的常见声音趋势**
       - **少年向**: 热血战斗类通常音量动态更大、节奏更快、爆发感更强；校园日常类则更自然松弛，角色差异更多体现在能量感和吐字方式上。
       - **少女向**: 校园恋爱类常见更细腻的情绪收放、更多气声与停顿；偶像/魔法少女类更强调甜美、元气、节奏感和感染力。
       - **青年向**: 悬疑/推理/阵营战斗类常更沉稳克制，语速平稳，信息表达清晰；奇幻/冒险类则更看角色身份差异，如战士的力量感、法师的神秘感、王者的威严感。
       - **女性向**: 职场/现实题材更贴近日常成熟表达，讲究职业感与真实感；古风/宫廷题材则可结合文雅措辞、仪式感和节制的情绪控制来刻画。

    **5. 常见原型映射（可优先参考，不是强制标签）**
       - **元气少女**: 音色清亮偏甜，音高偏高，语速快，能量感强，句尾轻快上扬较常见；若极度兴奋或受挫导致节奏突然失衡，优先写进 `content`。
       - **热血少年**: 声音扎实有冲劲，语速快，音量动态大，常有用力发声或短促呐喊；真正的嘶喊、喘息和极限爆发写进 `content`。
       - **傲娇**: 面对在意对象时常出现强硬外壳与真实情绪的反差，可能体现为语速变快、句尾走势多变、口头上不耐烦但气息不稳。
       - **病娇**: 日常态可能温柔亲近、气息轻柔，黑化态则会明显变冷、变慢、压抑；关键是两种状态的落差，而不是单纯“阴森”。
       - **天然呆**: 音色柔和干净，语速偏慢，停顿稍长，疑问和困惑感明显，整体有天然迟滞感。
       - **三无/无口**: 音量低、语速稳定、情绪外露少、语调起伏小；若在关键时刻第一次出现轻微颤抖或情绪松动，应明确写入 `content`。
       - **冷艳御姐**: 音色成熟偏低，语速适中偏慢，控制力强，句尾多下沉或平稳，常带距离感和压迫感。
       - **腹黑**: 表面温和亲切，内里却可能在重音、笑意质感、句尾停顿或语气温度上透出算计感，关键是“温柔外表下的冷意”。
       - **大叔/导师型**: 音色沉稳厚实，语速平缓，停顿有分量，整体更耐心、有说服力，适合承载教导感或阅历感。
       - **萝莉/正太**: 音色稚嫩、节奏不稳、好奇心强，语调偏上扬；哭闹、撒娇或突然慌张等临时情绪变化优先写入 `content`。
       - **智性型**: 音色清晰冷静，节奏稳定，逻辑感强，情绪波动低；少量罕见的情绪抬升更值得在 `content` 中强调。
    """
).strip()

ANIME_VOICE_ARCHETYPE_SECTION = dedent(
    """
    **6. 少儿拟人动画与阵营战斗动画的声音原型矩阵**
       - 这类作品常把角色塑造成高度拟人化的群像：你应优先总结**声音气质、表演节奏和叙事功能**，而不是总结外形原型。
       - 负例：`Subject1: 某种非人角色，战士型，声音低沉。`
       - 改写：`Subject1: 青年男性，带有热血战士感，音色偏硬朗，语速中快，命令感和冲锋感都很强。`

    **7. 三组高频声音原型**
       - **少儿群像喜剧 / 明快冒险型**: 常见于明亮、快节奏、儿童向或科普冒险类动画。重点关注音色是否明亮、语速是否偏快、反应是否灵活、情绪是否外显、整体是否带童趣、机灵感和一点轻微夸张感。
         - 可优先使用：`机灵少年主角感`、`元气冒险者感`、`聪明小领队感`、`明快群像角色感`、`知识讲述型少年角色感`。
       - **失败者喜剧反派 / 家庭感反派型**: 常见于反复吃瘪、嘴硬逞强、带家庭关系或日常喜剧色彩的反派角色。重点关注成年男性中低音、戏剧化停顿、爱把话说满、情绪外露、抱怨感和翻车后的泄气感。
         - 可优先使用：`失败者喜剧反派感`、`嘴硬爱逞强的成年反派感`、`家庭感喜剧反派`、`滑稽配角感`。
       - **热血战斗 / 阵营统帅型**: 常见于阵营对抗、战斗口号、热血冲锋和统率关系明确的动画。重点关注硬朗发声、低中音力量感、命令感、口号感、果断节奏，以及先锋型、统帅型、谋略型、强敌型之间的差别。
         - 可优先使用：`热血战士感`、`速度型先锋感`、`沉稳城主/导师感`、`高傲强敌感`、`阴狠谋略型反派感`。

    **8. 写法禁则**
       - 不要因为 scene、tag 或文本暗示这是强拟人化动画，就把 `speakers` 写成具体物种名、材质名、载具名或外形识别结果。
       - 不要把角色的“非人设定”直接等同于“电子音”“AI音感”或其他强处理的非自然音色；只有在音频里真实听得到合成、声码器、重处理或明显非自然发声时，才可写相关描述。
       - 这些原型词必须继续落在听觉证据上，例如音色亮或厚、节奏快或稳、情绪外显或克制、吐字利落或拖腔、命令感、抱怨感、打趣感、讲解感等。
    """
).strip()

ANIME_GUOFENG_SECTION = dedent(
    """
    **9. 国漫/古风专项**
       - 若角色稳定使用文雅措辞、敬语、书面表达、法诀式念白、轻微戏腔感或押韵感，可写入 `speakers` 的稳定风格。
       - 若角色稳定带有某种轻微口音、门派感或地域感，也可写入 `speakers`；但只出现一次的模仿式方言、夸张腔调或搞笑变声，应写入 `content`。
       - 像“莫欺少年穷”“天不生我某某，万古如长夜”这类强角色化口头语，只在它们稳定出现并真正构成角色辨识度时，才可作为辅助证据写入 `speakers`。
    """
).strip()

ANIME_VILLAIN_SECTION = dedent(
    """
    **10. 反派专项：强调类型差异，避免刻板化**
       - **优雅型反派**: 通常不是靠大音量取胜，而是靠从容语速、过分冷静的语调、压迫性的停顿和轻微冷笑制造威胁感。
       - **狂气型反派**: 关键在平静与癫狂之间的快速切换，而不是全程大喊大叫；笑声、语速和音量的剧烈波动是重要线索。
       - **阴柔型反派**: 可能体现为偏细或偏黏的声线、拖长尾音、气声、玩味口吻，但不要把这类角色简化成刻板“怪腔怪调”；真正的威胁感往往来自温柔音色与恶毒意图的反差。
    """
).strip()

ANIME_LOCAL_ALIGNMENT_SECTION = dedent(
    """
    **11. `speakers` 与 `content` 的分工对齐**
       - `speakers` 只写稳定基线，不要把“这一句突然更响、更快、更激动”直接写成长期风格。
       - `content` 用于记录单句内的临时变化，例如音量爆发、语速骤快、声线发抖、情绪翻脸、结巴、气声增多、突然冷下来、短促笑声质感变化等。
       - 负例：`speakers` 写“青年男性，时而愤怒时而悲伤”，`content` 只写“他很生气”。
       - 改写：`speakers` 写“青年男性，音色明亮，常态语速适中，带有少年感”；`content` 写“Subject1厉声喊道<S1>为什么！</S1> 这一句声音几乎冲到峰值，带着失控的愤怒。”或“Subject1低声说道<S1>对不起。</S1> 他后半句明显慢了下来，气声变多，情绪迅速沉下去。”
    """
).strip()

ANIME_EXAMPLES_SECTION = dedent(
    """
    **12. 参考案例**
       - **少儿机灵主角**
         - **Speakers**: "Subject1: 少年男性，带有机灵少年主角感，音色明亮，语速偏快，反应很快，吐字清楚，整体有轻巧的冒险感和稳定的元气感。"
         - **Local**: "Subject1快速解释道：<S1>先别慌，我们换个办法试试！</S1> 他这一句几乎是一口气冲出来的，前半句抢得很快，带着临场想出主意时的兴奋感。"
       - **喜剧反派**
         - **Speakers**: "Subject1: 中年男性，带有失败者喜剧反派感，音色中低，语气爱逞强，常把话说得很满，情绪起伏明显，带一点夸张的戏剧感和家庭化的抱怨感。"
         - **Local**: "Subject1气急败坏地喊道：<S1>这次我明明就差一点点！</S1> 他前半句还在硬撑，后半句却明显泄了气，抱怨味一下子更重了。"
       - **热血战士与统帅型角色**
         - **Speakers**: "Subject1: 青年男性，带有热血战士感，音色偏硬朗，语速中快，命令感和冲锋感都很强。Subject2: 中年男性，带有沉稳统帅感，音色偏低，语速稳，停顿有分量，整体判断更冷静。"
         - **Local**: "Subject1大声号令：<S1>所有人跟我上！</S1> 这一句冲锋感极强，起句就把音量抬了起来。随后Subject2沉声补充：<S2>别乱阵脚，按计划推进。</S2> 他语速不快，但每个重音都压得很稳。"
       - **热血少年对战**
         - **Speakers**: "Subject1: 少年，热血主角型，音色清亮有冲劲，语速快，音量动态大，发声偏用力，常态情绪外露。Subject2: 青年男性，宿敌型角色，音色偏低，语速沉稳，语调克制，带有持续压迫感。"
         - **Local**: "Subject1大吼道：<S1>这一招，是为你准备的！</S1> 他后半句几乎是顶着极限音量喊出来的，带着不顾一切的爆发感。"
       - **傲娇对话**
         - **Speakers**: "Subject1: 少女，带有傲娇角色感，音色偏亮，吐字利落，语速中快，面对在意对象时句尾走势多变，表面强硬但情绪藏不住。Subject2: 青年男性，音色温和，语速平缓，整体带有安抚和包容感。"
         - **Local**: "Subject1扭过脸说道：<S1>我、我才不是担心你呢！</S1> 她前半句还在逞强，句尾却明显变轻变快，露出掩不住的慌乱。"
       - **古风权谋**
         - **Speakers**: "Subject1: 老年男性，权臣角色，音色沙哑偏沉，语速缓慢，吐字极稳，带有文雅但危险的算计感。Subject2: 青年男性，君主角色，声音明亮但控制得略显吃力，试图维持威严。"
         - **Local**: "Subject1微微欠身说道：<S1>陛下，老臣愚钝。</S1> 他整句表面恭敬，但后半句刻意放轻，讥讽意味很重。"
       - **优雅反派**
         - **Speakers**: "Subject1: 青年男性，优雅型反派，音色华丽偏低，语速从容，吐字像戏剧念白般清晰，常带若有若无的冷笑。"
         - **Local**: "Subject1轻轻笑了一声后说道：<S1>真是，一场无聊的闹剧。</S1> 他把“无聊”两个字拖得很轻，反而更显轻蔑。"
       - **三无角色情绪解冻**
         - **Speakers**: "Subject1: 少女，三无/无口角色，音量低，语速恒定，语调起伏极小，音色偏冷，常态情绪高度克制。"
         - **Local**: "看到旧照片后，Subject1轻声说道：<S1>哥哥...</S1> 这一次她的句尾第一次出现了微弱松动，声音里有几乎听不清的颤抖。"
    """
).strip()

ANIME_GUIDANCE = "\n\n".join(
    (
        ANIME_GUIDANCE_OVERVIEW,
        ANIME_STYLE_MATRIX_SECTION,
        ANIME_VOICE_ARCHETYPE_SECTION,
        ANIME_GUOFENG_SECTION,
        ANIME_VILLAIN_SECTION,
        ANIME_LOCAL_ALIGNMENT_SECTION,
        ANIME_EXAMPLES_SECTION,
    )
)

ANIME_COMPACT_GUIDANCE = dedent(
    """
    此为动漫/动画场景，请优先刻画**稳定的配音表演基线**，所有判断都必须基于声音证据。

    **1. `speakers` 写法**
       - 顺序保持：`性别` > `年龄段` > `角色声音原型` > `稳定音色` > `常态说话风格`。
       - `speakers` 只写稳定基线；单句内突然变响、变快、翻脸、发抖、气声增多等变化优先写进 `content`。
       - 对强拟人化动画，也优先写**人格化的声音原型**，不要把 `speakers` 写成物种名、材质名、外形识别结果或“机械音/电子音”，除非音频里真的有明显合成处理。

    **2. 高频声音原型**
       - 少儿群像/明快冒险：关注明亮音色、偏快语速、反应快、情绪外显、童趣与机灵感。可用：`机灵少年主角感`、`元气冒险者感`、`聪明小领队感`、`知识讲述型少年角色感`。
       - 失败者喜剧反派：关注中低音、嘴硬逞强、爱把话说满、抱怨感、戏剧化停顿和翻车后的泄气感。可用：`失败者喜剧反派感`、`嘴硬爱逞强的成年反派感`、`家庭感喜剧反派`、`滑稽配角感`。
       - 热血战斗/阵营统帅：关注硬朗发声、低中音力量感、口号感、命令感、果断节奏，以及先锋型、统帅型、谋略型、强敌型差别。可用：`热血战士感`、`速度型先锋感`、`沉稳城主/导师感`、`高傲强敌感`、`阴狠谋略型反派感`。

    **3. 额外规则**
       - 国漫/古风可补文雅措辞、书面表达、轻微戏腔、稳定口音，但只在稳定存在时写入 `speakers`。
       - 优雅型、狂气型、阴柔型反派要强调类型差异，避免刻板化。
       - 若证据不足，宁可保守写“带有少年感/冷淡感/元气感”，不要强行写得过细。
    """
).strip()

SHORT_DRAMA_COMPACT_GUIDANCE = dedent(
    """
    此为短剧/微短剧场景，可结合剧情逻辑刻画更鲜明的角色身份，但必须以音频和文本证据为准。

    **1. `speakers` 写法**
       - 顺序保持：`性别` > `年龄段` > `角色画像/身份` > `稳定音色` > `常态说话风格`。
       - 身份和关系只用于**实际发声者**，不能为未开口角色单独创建条目。
       - `speakers` 写稳定画像；单句内的爆发、语速骤快、情绪失控、突然谄媚或突然变冷，优先写进 `content`。

    **2. 可优先关注**
       - 古风：文雅措辞、仪式感、轻微戏腔。
       - 都市/职场：吐字清晰、节奏较快、专业感和逻辑感。
       - 甜宠/反差：在不同对象前风格反差明显。
       - 若证据足够，可使用更具体的人设画像，如高冷总裁、干练御姐、腹黑反派、话痨角色等，但要放在性别年龄之后。

    **3. 风格细化**
       - 主动关注空灵感、谄媚感、油滑感、妖异感、阴柔感、刻薄感、狗腿感、卑微讨好感等更细的风格词。
       - 稳定贯穿的写进 `speakers`，只在单句短暂出现的写进 `content`。
    """
).strip()

ADVERTISING_COMPACT_GUIDANCE = dedent(
    """
    此为广告/数字人场景，请把 `speakers` 尽量对齐到给定知识库，并保持“性别 + 年龄”写在最前面。

    **1. `speakers` 写法**
       - 推荐顺序：`性别` > `年龄` > `人设大类` > `声音风格大类` > `音色特点` > `稳定口播方式`。
       - 尽量从知识库里选择最匹配的人设、风格和音色，但证据不足时宁可保守，不要为了凑标签而臆造。

    **2. 优先识别**
       - 人设大类：如带货主播、好物分享官、金融讲师、教育从业者、商务精英、数码产品推荐、养生理疗师、都市丽人等。
       - 声音风格大类：如通用广告风格、激情带货风格、娓娓道来养生风格、自然对话社交风格、故事感讲述风格、情景演绎风格。
       - 音色特点：如温柔亲和、干练利落、低沉沉稳、甜美灵动、醇厚磁性、清亮通透等。

    **3. `content` 职责**
       - `speakers` 只写稳定口播基线。
       - 突然加快、强调价格、制造紧迫感、引导下单、刻意压低声线、增强聊天感等单句策略变化，优先写入 `content`。
    """
).strip()

SCENE_PROMPT_RULES = (
    {
        "name": "anime",
        "display_name": "动漫/动画",
        "keywords": (
            "动漫",
            "动画",
            "anime",
            "cartoon",
            "二次元",
            "国漫",
            "日漫",
            "番剧",
            "配音",
        ),
        "guidance": ANIME_GUIDANCE,
        "compact_guidance": ANIME_COMPACT_GUIDANCE,
    },
    {
        "name": "short_drama",
        "display_name": "短剧/长剧/影视剧",
        "keywords": (
            "短剧",
            "微短剧",
            "短视频剧情",
            "竖屏短剧",
            "web drama",
            "micro drama",
            "长剧",
            "电视剧",
            "电影",
            "film",
            "tv drama",
        ),
        "guidance": (
            "此为短剧场景，允许结合剧情逻辑，刻画**更鲜明的角色身份**。\n\n"
            "**1. 描述顺序与`content`使用**\n"
            "   - **稳定描述顺序**: 每个 `speakers` 条目必须先写`性别`和`年龄段`，再补充`角色画像`、`稳定音色`、`常态说话风格`。不要把“高冷总裁”等身份标签直接放在最前面。\n"
            "   - **`content` 的作用**: 某一句里的临时变化，如音量爆发、语调抬升、语速突然加快、情绪激动等，应优先写到 `content` 中，以保持 `speakers` 风格基线的稳定。\n\n"
            "**2. 建立风格基线**\n"
            "   - 根据常见的短剧原型，预判语音特征。例如：\n"
            "       *   **古风**: 语速偏慢，措辞文雅，可能带有轻微的戏腔感或仪式感。\n"
            "       *   **都市/职场**: 语速较快，吐字清晰，对话专业且有逻辑感。\n"
            "       *   **甜宠/反差**: 角色音色、语速、情绪在不同对象前呈现巨大反差，应在 `speakers` 中注明这种多面性。\n\n"
            "**3. 补充角色身份**\n"
            "   - 当台词、称谓和互动方式提供足够线索时，可在 `speakers` 中加入角色身份或关系，如“老板与秘书”、“公主与侍卫”、“总裁与夫人”。\n"
            "   - **注意**: 这些身份只能用于**实际发声**的 speaker，不能为未开口的角色单独创建条目。\n\n"
            "**4. 使用具体人设画像**\n"
            "   - 当声音表演、说话方式、称谓关系和文本线索都**高度一致**时，可以在 `speakers` 中使用更具体的人设画像，如“娇蛮公主、温润男主、高冷总裁、干练御姐、腹黑反派、阳光少年、元气少女、耍宝役、话痨角色”等。\n"
            "   - 此类画像应放在`性别` `年龄`之后，作为风格补充，而不是替代基础声音描述。\n\n"
            "**5. 关注中性/非典型角色**\n"
            "   - 对短剧里常见的非典型角色也要提高敏感度，如“小丑感、太监感、雌雄莫辨的中性角色、妖异角色、滑稽丑角、阴柔反派”等。\n"
            "   - 如果角色的性别气质明显偏中性、阴柔或难以仅凭音频直接判断，也要如实描述其声音呈现，例如“中性气质明显、声线偏尖细、拿腔拿调、尾音拖长、语气刻意讨好”等。\n\n"
            "**6. 细化风格词**\n"
            "   - 除了常见的霸道、威严、热血，也要主动关注更细的风格标签，如“空灵感、谄媚感、油滑感、滑稽感、妖异感、阴柔感、刻薄感、狗腿感、卑微讨好感、拿腔作势感、故作娇弱感”等。\n"
            "   - 若风格稳定贯穿，写进 `speakers`；若只在单句短暂出现，则写进 `content`。\n\n"
            "**7. 提供听觉证据**\n"
            "   - 所有身份和风格判断必须有声音特征支撑。例如：\n"
            "       *   “**秘书**”的专业感体现在“语速快、吐字清晰、用词正式”。\n"
            "       *   “**侍卫**”的忠诚感体现在“声音沉稳、应答简短有力”。\n"
            "       *   “**太监感/谄媚感**”可能体现在“声线偏细、笑意刻意、尾音上挑、措辞过分恭敬、语气试探”。\n"
            "       *   “**空灵感**”可能体现在“气声较多、共鸣轻、语流飘逸、音高偏高但不刺耳”。\n"
            "   - 通过“语气中的权力距离”、“称谓用法”、“情绪收放方式”、“是否刻意讨好或故作姿态”等听觉证据来增强角色鲜明度。\n\n"
            "**短案例**\n\n"
            "   - **案例1 (都市霸总)**\n"
            '       *   **Speakers**: "Subject1: 青年男性，带有高冷总裁式角色画像，声音低沉且带有命令感，语速快，音量偏大，压迫感强。Subject2: 青年女性，带有下属角色感，声音偏高且带有一丝紧张，语速快，音量略轻，回应简短。"\n'
            '       *   **BGM**: "节奏感强的电子音乐，用于营造紧张的对峙氛围。"\n\n'
            "   - **案例2 (古风权谋)**\n"
            '       *   **Speakers**: "Subject1: 老年男性，带有腹黑反派式权臣画像，声音沙哑，语速缓慢，音量克制，充满算计的语气。Subject2: 青年男性，带有年轻君主角色感，声音明亮但气息不稳，音量偶有波动，显示出其年轻和经验不足。"\n'
            '       *   **Env**: "空旷的大殿，有明显回声，背景有<Audio>烛火燃烧的噼啪声</Audio>。"\n\n'
            "   - **案例3 (宫廷谄媚角色)**\n"
            '       *   **Speakers**: "Subject1: 中年男性，带有太监/近侍式角色画像，声线偏细，语气谄媚，尾音常轻微上挑，措辞过分恭敬，整体透出卑微讨好的风格。"\n'
            '       *   **Local**: "Subject1陪笑着低声禀报道：<S1>奴才这就去给您安排。</S1>"\n\n'
            "   - **案例4 (空灵妖异角色)**\n"
            '       *   **Speakers**: "Subject1: 青年女性，带有空灵妖异式角色画像，声线轻飘、气声较多，语速偏缓，语调游离，整体有不真实的空灵感。"'
        ),
        "compact_guidance": SHORT_DRAMA_COMPACT_GUIDANCE,
    },
    {
        "name": "advertising",
        "display_name": "广告/数字人",
        "keywords": (
            "广告",
            "口播",
            "数字人",
            "数字人口播",
            "卖货",
            "带货",
            "种草",
            "直播带货",
            "直播口播",
            "商品推荐",
            "好物分享",
            "营销",
            "商单",
            "promotion",
            "advertisement",
            "ad",
            "commercial",
        ),
        "guidance": (
            "此为广告/数字人场景。请把 `speakers` 的描述尽量对齐到给定知识库，从**人设大类**、**声音风格大类**、**音色特点**、**年龄**四个维度中优先选择最匹配的标签，并保持“性别 + 年龄”写在最前面。\n\n"
            "**1. `speakers` 的推荐描述顺序**\n"
            "   - 每个说话人优先按以下顺序描述：`性别` > `年龄` > `人设大类` > `声音风格大类` > `音色特点` > `稳定说话方式/职业感/口播特征`。\n"
            "   - 其中“性别 + 年龄”必须写在最前面，例如：`青年女性，20-30岁，都市丽人...`、`中年男性，40-50岁，商务精英...`。\n"
            "   - 如果证据足够，尽量同时覆盖这四个维度；如果某一维证据不足，可少写，但不要臆造。\n\n"
            "**2. 人设大类知识库（优先从中选择）**\n"
            "   - 受访者、医疗人员、教育从业者、宝爸宝妈、金融讲师、商务精英、邻家素人、专业演讲者、权威科研演员人设者、中医从业者、好物分享官、数码产品推荐、数码维修者、户外感悟分享者、都市丽人、民族传承人、股市专家、国学讲师、老年教育家、校园演讲者、装修员工、快递员、主持人、工人、农民、司机、销售员、工程师、保姆、厨师、外卖员、保洁员、客服、运动从业者、程序员、带货主播、养生理疗师、游戏主播、律师、摄影师、美容师、超模、空姐、空少、服务员、其他。\n"
            "   - 这些标签适合写在“年龄”之后，作为该说话人的职业/身份/数字人设定。\n"
            "   - 只有当声音表现、措辞、口播结构、职业术语或语气习惯足够支持时，才明确归入某一类；若证据不足，可写“其他”或只写更保守的基础画像。\n\n"
            "**3. 声音风格大类（优先从中选择）**\n"
            "   - 通用广告(可分享、可卖货)风格、激情带货风格、娓娓道来养生风格、特色吸睛风格、自然对话社交风格、故事感讲述风格、情景演绎风格、其他风格。\n"
            "   - 这是对该 speaker 在整段音频中的**稳定口播风格**概括，建议尽量补到 `speakers` 里，而不是只写在 `content`。\n"
            "   - 如果只是某一句突然提高音量、突然更激动，这类临时变化仍应优先写入 `content`。\n\n"
            "**4. 音色特点（优先从中选择）**\n"
            "   - 温柔亲和、甜美灵动、娇俏可爱、御姐、热情大气、元气阳光、醇厚磁性、幽默风趣、干练利落、沙哑慵懒、清亮通透、低沉沉稳、沧桑磁性、软糯清甜、铿锵有力、清冷疏离、沉稳随和。\n"
            "   - 这是对 speaker 稳定音色气质的提炼，建议优先从上述集合中选 1-3 个最贴切的词，再用可听见的证据展开。\n"
            "   - 不要只堆标签，最好补一句对应的听觉支撑，例如：音色偏亮、吐字利落、音量外放、语速偏快、气息稳等。\n\n"
            "**5. 年龄标签（优先从中选择）**\n"
            "   - 3岁以下、10岁以下、20岁以下、20-30岁、30-40岁、40-50岁、50岁以上。\n"
            "   - 年龄必须写在最前面，与性别一起出现。\n"
            "   - 若只能判断大致成年/中年/老年，也尽量映射到最接近的年龄段，而不是写非常模糊的年龄描述。\n\n"
            "**6. 广告/数字人口播的特别要求**\n"
            "   - 请优先识别该 speaker 属于哪种广告角色：例如带货主播、好物分享官、主持人、金融讲师、养生理疗师、数码产品推荐、都市丽人、邻家素人等。\n"
            "   - 再进一步判断其更接近哪种声音风格：例如通用广告风格、激情带货风格、自然对话社交风格、故事感讲述风格等。\n"
            "   - 最后再结合音色特点补全 speaker 画像，例如温柔亲和、干练利落、低沉沉稳、甜美灵动、醇厚磁性等。\n"
            "   - 对于数字人口播，若声音明显具有过度平滑、稳定度异常高、停顿整齐、情绪变化受控、广告结构强等特征，可在稳定风格里适度体现其“数字人口播感”或“强口播脚本感”。\n\n"
            "**7. `content` 的职责**\n"
            "   - `speakers` 只写稳定的人设与风格基线。\n"
            "   - 某一句中的临时变化，例如突然加快语速、刻意强调价格、语气猛然抬高、情绪突然更兴奋、结尾压低声线制造悬念等，应写入 `content`。\n"
            "   - 广告口播常见的“强调卖点”“制造紧迫感”“引导下单”“模仿聊天感”这类单句策略变化，也优先写入 `content`。\n\n"
            "**8. 写法要求**\n"
            "   - `speakers` 中尽量从以上词库中选择标签，但不能只做标签拼接，必须补上与声音一致的听觉描述。\n"
            "   - 如果能判断出人设大类、风格大类、音色特点、年龄这四个维度，尽量全写；若证据不足，则只保留最稳的维度。\n"
            "   - 不要为了凑标签而写不确定信息；不确定时宁可保守。\n\n"
            "**示例1（好物分享官 / 自然社交风格）**\n"
            '   - **Speakers**: "Subject1: 青年女性，20-30岁，好物分享官，自然对话社交风格，音色温柔亲和、清亮通透，整体说话方式像在和熟人聊天，吐字清晰，语气轻松。"\n'
            '   - **Local**: "Subject1像朋友聊天一样介绍道：<S1>这个真的不是智商税，我自己已经回购三次了。</S1> 说到回购次数时，她明显加重了语气来强调可信度。"\n\n'
            "**示例2（带货主播 / 激情带货风格）**\n"
            '   - **Speakers**: "Subject1: 中年女性，30-40岁，带货主播，激情带货风格，音色热情大气、铿锵有力，常态语速偏快，音量外放，煽动性强。"\n'
            '   - **Local**: "Subject1高能量地催促道：<S1>今天这个价格你们现在不拍，真的明天就没有了！</S1> 她后半句明显更响更快，制造出强烈的紧迫感。"\n\n'
            "**示例3（养生理疗师 / 娓娓道来风格）**\n"
            '   - **Speakers**: "Subject1: 中年男性，40-50岁，养生理疗师，娓娓道来养生风格，音色低沉沉稳、沉稳随和，语速适中偏慢，讲解感强。"\n'
            '   - **Local**: "Subject1平和地解释道：<S1>晚上总是睡不好的人，可以先试着把这两个习惯改掉。</S1> 他整句都很稳，像在耐心做长期建议。"\n\n'
            "**示例4（数码产品推荐 / 干练利落风格）**\n"
            '   - **Speakers**: "Subject1: 青年男性，20-30岁，数码产品推荐，通用广告(可分享、可卖货)风格，音色干练利落、清亮通透，吐字快而清楚，信息密度高。"\n'
            '   - **Local**: "Subject1迅速总结卖点：<S1>它最强的不是性能，而是这个价位里少见的影像稳定性。</S1> 说到核心卖点时，他刻意放慢了一点来突出重点。"\n\n'
            "**示例5（主持人 / 特色吸睛风格）**\n"
            '   - **Speakers**: "Subject1: 青年女性，20-30岁，主持人，特色吸睛风格，音色甜美灵动、元气阳光，开场感染力强，整体带有明显的节目感与吸睛感。"\n'
            '   - **Local**: "Subject1用抓人的开场语气说道：<S1>你敢相信吗，这个东西居然把我最头疼的问题一次解决了。</S1> 她开头的重音非常明显，明显是在制造注意力。"'
        ),
        "compact_guidance": ADVERTISING_COMPACT_GUIDANCE,
    },
)

AD_SCENARIO_SPLIT_PATTERN = re.compile(r"[-_/|｜,，、\s]+")

AD_SCENARIO_PERFORMANCE_HINTS = (
    {
        "keywords": ("高表现力", "强表现力", "强感染力", "高能量", "高张力", "强情绪"),
        "guidance": (
            "若广告细分标签强调“高表现力/高能量”，请提高对表达强度的关注，"
            "优先观察是否存在更强的感染力、重音更明显、口播推进感更强、卖点强调更外放的特征。"
            "这类信息更适合帮助你判断 `声音风格大类` 与 `content` 中的单句策略，但仍要以音频证据为准。"
        ),
    },
    {
        "keywords": ("低表现力", "低能量", "平静", "克制", "自然", "轻表达"),
        "guidance": (
            "若广告细分标签强调“自然/克制/低表现力”，请优先关注更轻松、更生活化、更像日常分享的口播方式，"
            "不要因为是广告场景就强行写成激情带货风格。"
        ),
    },
    {
        "keywords": ("专业", "专家", "权威", "讲解", "科普"),
        "guidance": (
            "若广告细分标签强调“专业/专家/讲解/科普”，请优先判断声音是否更接近医疗人员、教育从业者、金融讲师、律师、"
            "权威科研演员人设者等专业型人设，并关注其稳定的讲解感、条理感和可信度。"
        ),
    },
)

AD_SCENARIO_COMBINATION_HINTS = (
    {
        "required_keywords": ("高表现力", "电商"),
        "guidance": (
            "若广告细分标签是“高表现力-电商”，请优先关注 speaker 是否更接近带货主播、好物分享官、销售员、邻家素人等高转化口播人设。"
            "在 `声音风格大类` 上，优先判断是否更接近激情带货风格、特色吸睛风格或通用广告(可分享、可卖货)风格；"
            "在 `content` 中要更敏感地捕捉价格强调、限时催单、卖点连发、反问吸引、强节奏下单引导等单句策略。"
        ),
    },
    {
        "required_keywords": ("高表现力", "家居建材"),
        "guidance": (
            "若广告细分标签是“高表现力-家居建材”，请优先关注 speaker 是否更接近装修员工、销售员、好物分享官、邻家素人等家装口播人设。"
            "在 `speakers` 中可以更注意是否带有明显的讲解感、方案感、施工/材质/空间改造相关的话术习惯；"
            "在 `content` 中要更关注对痛点、材质卖点、耐用性、安全性、收纳改造、空间利用、价格对比等信息的高强度强调。"
        ),
    },
    {
        "required_keywords": ("高表现力", "健康"),
        "guidance": (
            "若广告细分标签是“高表现力-健康”，请优先关注 speaker 是否更接近医疗人员、中医从业者、养生理疗师、好物分享官等人设。"
            "在 `声音风格大类` 上重点区分是更强转化的特色吸睛风格/激情带货风格，还是带有稳定专业感的娓娓道来养生风格；"
            "在 `content` 中更关注对身体困扰、作息问题、改善效果、使用感受、健康建议等内容的强调方式和带动感。"
        ),
    },
    {
        "required_keywords": ("高表现力", "教育"),
        "guidance": (
            "若广告细分标签是“高表现力-教育”，请优先关注 speaker 是否更接近教育从业者、校园演讲者、国学讲师、老年教育家、专业演讲者等人设。"
            "在 `speakers` 中要更关注其讲解感、启发感、感染力和条理推进感；"
            "在 `content` 中更留意知识点强调、提问式吸引、成绩/成长痛点放大、报名转化、激励式结尾等广告化策略。"
        ),
    },
    {
        "required_keywords": ("高表现力", "金融"),
        "guidance": (
            "若广告细分标签是“高表现力-金融”，请优先关注 speaker 是否更接近金融讲师、股市专家、商务精英、专业演讲者等人设。"
            "在 `声音风格大类` 上可重点判断其更像特色吸睛风格、故事感讲述风格还是更强权威讲解感的通用广告风格；"
            "在 `content` 中更关注机会窗口、风险提醒、收益对比、专业术语密度、权威背书感和行动号召的表达方式。"
        ),
    },
    {
        "required_keywords": ("高表现力", "商务服务"),
        "guidance": (
            "若广告细分标签是“高表现力-商务服务”，请优先关注 speaker 是否更接近商务精英、专业演讲者、律师、工程师、客服等偏方案型或服务型人设。"
            "在 `speakers` 中可更关注其专业可信感、效率感、成交导向和商务表达习惯；"
            "在 `content` 中更留意降本增效、方案价值、品牌可信、服务承诺、痛点解决和转化引导等单句策略。"
        ),
    },
)

AD_SCENARIO_DOMAIN_HINTS = (
    {
        "keywords": ("健康", "养生", "理疗", "医疗", "中医"),
        "guidance": (
            "若广告细分标签落在健康/养生方向，请优先关注 speaker 是否更接近"
            "医疗人员、中医从业者、养生理疗师、好物分享官等人设；"
            "同时在 `声音风格大类` 上重点区分是“娓娓道来养生风格”、"
            "“自然对话社交风格”还是更强转化的“特色吸睛风格/激情带货风格”。"
        ),
    },
    {
        "keywords": ("数码", "科技", "电子", "家电", "手机", "电脑"),
        "guidance": (
            "若广告细分标签落在数码方向，请优先关注 speaker 是否更接近数码产品推荐、数码维修者、工程师等人设，"
            "并观察其表达是否更强调参数、功能拆解、效率感和信息密度。"
        ),
    },
    {
        "keywords": ("金融", "股市", "投资", "理财", "财经"),
        "guidance": (
            "若广告细分标签落在金融方向，请优先关注 speaker 是否更接近金融讲师、股市专家、商务精英等人设，"
            "并重点判断其权威感、节奏控制、专业术语密度和说服方式。"
        ),
    },
    {
        "keywords": ("教育", "学习", "课程", "培训", "知识"),
        "guidance": (
            "若广告细分标签落在教育方向，请优先关注 speaker 是否更接近教育从业者、国学讲师、校园演讲者、老年教育家等人设，"
            "并判断其是否带有明显的讲解感、启发感和条理化表达。"
        ),
    },
    {
        "keywords": ("美妆", "护肤", "彩妆", "香氛", "穿搭"),
        "guidance": (
            "若广告细分标签落在美妆方向，请优先关注 speaker 是否更接近美容师、都市丽人、好物分享官等人设，"
            "并观察其音色气质是否更偏甜美灵动、温柔亲和、干练利落或特色吸睛。"
        ),
    },
    {
        "keywords": ("母婴", "育儿", "宝宝", "宝妈", "宝爸"),
        "guidance": (
            "若广告细分标签落在母婴方向，请优先关注 speaker 是否更接近宝爸宝妈、邻家素人、好物分享官等人设，"
            "并判断其语气是否更像经验分享、安抚建议或照护场景下的自然口播。"
        ),
    },
)


TEXT_GUIDED_VOXCAPS_AUDIO_TEMPLATE = {
    "system_prompt": VOXCAPS_AUDIO_CAPTION_TEMPLATE["system_prompt"]
    + """

额外规则：
1. 用户可能会额外提供一段已知文本，这段文本可能来自独立 ASR 或 GT，通常比你直接听音频得到的转写更可靠。
2. 当已知文本存在时，`content` 中所有 `<S{id}>...</S{id}>` 内的语音文本内容必须优先参考这段已知文本。
3. 你仍然必须通过音频判断 `environment`、`speakers`、说话情绪、音效和事件顺序。
4. 不要输出额外字段，不要把“参考文本”原样单独作为新 key 输出；最终仍然只输出规定的 JSON 结构。
""",
    "user_prompt": VOXCAPS_AUDIO_CAPTION_TEMPLATE["user_prompt"]
    + f"""

### 4.1 已知文本约束（仅当提供时生效）
下面给你一段更可信的文本，它可能来自独立 ASR 或 GT：

<REFERENCE_TEXT>
{TEXT_PROMPT_PLACEHOLDER}
</REFERENCE_TEXT>

请严格遵守：
1. `content` 中的语音内容优先使用这段文本，不要随意改写、翻译或补写。
2. 你仍然要根据音频判断是谁在说、怎么说、何时说，以及同时发生了什么音效。
3. `environment`、`speakers` 不能只根据文本生成，仍然必须以音频为准。
4. 如果文本和音频在细枝末节上略有冲突，优先保留文本内容，并用音频补足环境与风格信息。
""",
}


ASR_DERIVED_TEXT_RULES = """
### 4.2 ASR 衍生文本的额外约束（仅当已知文本来自当前音频的自动转写时生效）
请特别注意：
1. 这段 `REFERENCE_TEXT` 只代表这条音频的整体转写参考，不代表说话人切分结果，也不保证已经按发言轮次分段。
2. 如果音频中存在多人对话、插话、抢话或重叠说话，`content` 中 `<S{id}>...</S{id}>` 的归属、切句边界、说话顺序和语气归属，必须重新根据音频判断。
3. 不允许因为 `REFERENCE_TEXT` 是一整段连续文本，就把整段内容机械地归给同一个 Subject。
4. 若音频中的说话轮次和 `REFERENCE_TEXT` 的断句并不完全一致，优先以音频事实为准，再尽量参考文本词面内容。
5. `speakers`、`environment` 依然只能根据音频事实总结，不能从 ASR 文本里反推说话人数、身份或轮次。
""".strip()


def deduplicate_strings(values: Sequence[str]) -> List[str]:
    deduplicated: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def parse_lang_values(lang_value: Optional[str]) -> List[str]:
    if lang_value is None:
        return []

    normalized_values: List[str] = []
    seen = set()
    for raw_value in lang_value.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        normalized_value = LANGUAGE_NAME_MAP.get(raw_value.lower(), raw_value)
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized_values.append(normalized_value)
    return normalized_values


def build_language_prompt_hint(lang_values: Sequence[str]) -> Optional[str]:
    unique_lang_values = deduplicate_strings(lang_values)
    if not unique_lang_values:
        return None

    joined_langs = "、".join(unique_lang_values)
    if len(unique_lang_values) == 1:
        language_name = unique_lang_values[0]
        return (
            f"当前任务的语音语种固定为：{language_name}。\n"
            f"请确保 `content` 中所有 `<S{{id}}>...</S{{id}}>` 内的语音文本都使用{language_name}转写，"
            "并且与实际音频内容一致。\n"
            "不要把语音翻译成其他语言，也不要为了润色而改写说话内容。\n"
            "如果提供了已知文本（如 text/transcript/GT），也必须在尊重该文本的同时保持与音频语种一致。"
        )

    return (
        f"当前任务允许的语音语种范围为：{joined_langs}。\n"
        "请确保 `content` 中所有 `<S{id}>...</S{id}>` 内的语音文本保持音频原语种，并与实际音频内容一致。\n"
        f"只允许输出上述语种范围内的语音文本，不要翻译成其他语言，也不要输出不在该范围内的文本。\n"
        "如果同一条音频里实际说的是中文，就写中文；如果实际说的是英文，就写英文；"
        "若出现中英混说，也应按音频原样保留，不要统一翻译。"
    )


def normalize_ad_scenario_tokens(ad_scenario_values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for raw_value in ad_scenario_values:
        if not isinstance(raw_value, str):
            continue
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        for part in AD_SCENARIO_SPLIT_PATTERN.split(raw_value):
            part = part.strip()
            if part:
                tokens.append(part)
    return deduplicate_strings(tokens)


def build_advertising_scenario_prompt_hint(
    ad_scenario_values: Sequence[str],
    compact: bool = False,
) -> Optional[str]:
    unique_values = deduplicate_strings(ad_scenario_values)
    if not unique_values:
        return None

    scenario_tokens = normalize_ad_scenario_tokens(unique_values)
    normalized_value_text = " ".join(unique_values)
    if compact:
        hint_lines = [
            f"广告细分场景标签：{'、'.join(unique_values)}",
            "请仅把这些标签当作广告场景下的聚焦线索，用来细化 `speakers` 与 `content` 的判断重心，仍以音频事实为准。",
        ]
    else:
        hint_lines = [
            f"当前广告细分场景标签：{'、'.join(unique_values)}",
            "请把这些标签当作广告场景下的补充知识，用来细化 `speakers` 与 `content` 的判断重心。",
            "- 一般来说，类似 `高表现力-健康` 这样的标签可以拆成“表达强度/投放风格”与“行业内容方向”两部分理解。",
            "- 前半部分更适合帮助你判断 speaker 的广告风格强度、节奏感、感染力和 `content` 中的表达策略。",
            "- 后半部分更适合帮助你判断 speaker 更接近哪类广告人设、哪种行业话术，以及应优先从知识库里选择哪些人设大类和声音风格大类。",
            "- 这些标签只能帮助你聚焦判断，不能覆盖音频事实；如果音频证据不足，宁可保守，也不要强行套标签。",
        ]

    appended_hint_count = 0
    for rule in AD_SCENARIO_COMBINATION_HINTS:
        if all(keyword in normalized_value_text for keyword in rule["required_keywords"]):
            hint_lines.append(f"- {rule['guidance']}")
            appended_hint_count += 1
            if compact and appended_hint_count >= 3:
                break

    for rule in AD_SCENARIO_PERFORMANCE_HINTS:
        if any(keyword in token for token in scenario_tokens for keyword in rule["keywords"]):
            hint_lines.append(f"- {rule['guidance']}")
            appended_hint_count += 1
            if compact and appended_hint_count >= 4:
                break

    for rule in AD_SCENARIO_DOMAIN_HINTS:
        if any(keyword in token for token in scenario_tokens for keyword in rule["keywords"]):
            hint_lines.append(f"- {rule['guidance']}")
            appended_hint_count += 1
            if compact and appended_hint_count >= 5:
                break

    hint_lines.append(
        "- 如果细分标签和音频共同支持，你可以更积极地从广告知识库中选择“人设大类 + 声音风格大类 + 音色特点”；"
        "但 `speakers` 仍必须以“性别 + 年龄”开头。"
    )
    if compact:
        hint_lines.append(
            "- `content` 重点关注该细分场景下的单句策略，如强强调、耐心解释、参数卖点强化、转化引导等。"
        )
    else:
        hint_lines.append(
            "- `content` 中可以更关注与该广告细分场景相关的单句策略，例如健康类的耐心解释、"
            "高表现力类的强强调与强转化、数码类的参数卖点强化等。"
        )
    return "\n".join(hint_lines)


def build_scene_prompt_hint(
    scene_values: Sequence[str],
    ad_scenario_values: Optional[Sequence[str]] = None,
    compact: bool = False,
) -> Optional[str]:
    if not scene_values:
        return None

    matched_rules = []
    matched_rule_names = set()
    for raw_value in scene_values:
        normalized_value = raw_value.lower()
        for rule in SCENE_PROMPT_RULES:
            if rule["name"] in matched_rule_names:
                continue
            if any(keyword in normalized_value for keyword in rule["keywords"]):
                matched_rules.append(rule)
                matched_rule_names.add(rule["name"])

    if not matched_rules:
        return None

    unique_scene_values = "、".join(deduplicate_strings(scene_values))
    guidance_lines = [f"当前分类标签：{unique_scene_values}", "请额外遵循以下补充要求："]
    for rule in matched_rules:
        guidance_text = rule.get("compact_guidance") if compact else None
        if not guidance_text:
            guidance_text = rule["guidance"]
        guidance_lines.append(f"- {rule['display_name']}：{guidance_text}")
    if "advertising" in matched_rule_names:
        advertising_scenario_hint = build_advertising_scenario_prompt_hint(
            ad_scenario_values or [],
            compact=compact,
        )
        if advertising_scenario_hint:
            guidance_lines.append(f"- 广告细分场景补充：{advertising_scenario_hint}")
    if compact:
        guidance_lines.append(
            "- 若已知文本里有明确称谓、身份或关系，可作为辅助依据，但只在音频或文本足够支持时使用。"
        )
    else:
        guidance_lines.append(
            "- 如果已知文本中存在明确称谓、身份或角色关系，也可以将其作为辅助依据；"
            "但只在音频或文本足够支持时使用，避免臆造。"
        )
    return "\n".join(guidance_lines)


def build_voxcaps_audio_prompts(
    known_text: Optional[str],
    scene_prompt_hint: Optional[str],
    language_prompt_hint: Optional[str],
    known_text_source: str = "trusted",
) -> Tuple[str, str]:
    return _build_voxcaps_prompts(
        known_text=known_text,
        scene_prompt_hint=scene_prompt_hint,
        language_prompt_hint=language_prompt_hint,
        known_text_source=known_text_source,
        base_template=VOXCAPS_AUDIO_CAPTION_TEMPLATE,
        text_guided_template=TEXT_GUIDED_VOXCAPS_AUDIO_TEMPLATE,
    )


def _build_voxcaps_prompts(
    known_text: Optional[str],
    scene_prompt_hint: Optional[str],
    language_prompt_hint: Optional[str],
    known_text_source: str,
    base_template: dict,
    text_guided_template: dict,
) -> Tuple[str, str]:
    if not known_text:
        system_prompt, user_prompt = (
            base_template["system_prompt"],
            base_template["user_prompt"],
        )
    else:
        system_prompt = text_guided_template["system_prompt"]
        user_prompt = text_guided_template["user_prompt"].replace(
            TEXT_PROMPT_PLACEHOLDER,
            known_text,
        )

        if known_text_source == "asr":
            user_prompt = user_prompt + "\n\n" + ASR_DERIVED_TEXT_RULES

    if scene_prompt_hint:
        user_prompt = (
            user_prompt
            + "\n\n### 5. 场景化标注指导（仅当前样本生效）\n"
            + scene_prompt_hint
            + "\n\n请注意：这些指导只用于微调描述粒度和角色刻画方式，"
            + "不能覆盖音频事实，也不能改变最终 JSON 结构。"
        )

    if language_prompt_hint:
        user_prompt = (
            user_prompt + "\n\n### 6. 语种约束（当前任务全局生效）\n" + language_prompt_hint
        )

    return system_prompt, user_prompt
