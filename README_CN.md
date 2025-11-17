# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

杞婚噺鐗圡APPO锛屽府鍔╀綘蹇€熺Щ妞嶅埌鏈湴鐜銆?
- [瑙嗛瑙ｆ瀽](https://www.bilibili.com/video/BV1bd4y1L73N/?spm_id_from=333.999.0.0&vd_source=d8ab7686ea514acb6635faa5d2227d61)  

鑻辨枃缈昏瘧鐗坮eadme锛岃鐐瑰嚮[杩欓噷](README.md)

## Table of Contents

- [鑳屾櫙](#鑳屾櫙)
- [瀹夎](#瀹夎)
- [鐢ㄦ硶](#鐢ㄦ硶)

## 鑳屾櫙

MAPPO鍘熺増浠ｇ爜瀵逛簬鐜鐨勫皝瑁呰繃浜庡鏉傦紝鏈」鐩洿鎺ュ皢鐜灏佽鎶藉彇鍑烘潵銆傛洿鍔犳柟渚垮皢MAPPO浠ｇ爜绉绘鍒拌嚜宸辩殑椤圭洰涓娿€?
## 瀹夎

鐩存帴灏嗕唬鐮佷笅杞戒笅鏉ワ紝鍒涘缓涓€涓狢onda鐜锛岀劧鍚庤繍琛屼唬鐮侊紝缂哄暐琛ュ暐鍖呫€傚叿浣撲粈涔堝寘浠ュ悗鍐嶆坊鍔犮€?
## 鐢ㄦ硶

- 鐜閮ㄥ垎鏄竴涓┖鐨勭殑瀹炵幇锛屾枃浠禶light_mappo/envs/env_core.py`閲岄潰鐜閮ㄥ垎鐨勫疄鐜帮細[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # 鐜涓殑鏅鸿兘浣?    """
    def __init__(self):
        self.agent_num = 2  # 璁剧疆鏅鸿兘浣?灏忛鏈?鐨勪釜鏁帮紝杩欓噷璁剧疆涓轰袱涓?        self.obs_dim = 14  # 璁剧疆鏅鸿兘浣撶殑瑙傛祴缁村害
        self.action_dim = 5  # 璁剧疆鏅鸿兘浣撶殑鍔ㄤ綔缁村害锛岃繖閲屽亣瀹氫负涓€涓簲涓淮搴︾殑

    def reset(self):
        """
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝杩斿洖鍊间负涓€涓猯ist锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.obs_dim, )鐨勮娴嬫暟鎹?        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝actions鐨勮緭鍏ヤ负涓€涓?绾殑list锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.action_dim, )鐨勫姩浣滄暟鎹?        # 榛樿鍙傛暟鎯呭喌涓嬶紝杈撳叆涓轰竴涓猯ist锛岄噷闈㈠惈鏈変袱涓厓绱狅紝鍥犱负鍔ㄤ綔缁村害涓?锛屾墍閲屾瘡涓厓绱爏hape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```



# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

杞婚噺鐗圡APPO锛屽府鍔╀綘蹇€熺Щ妞嶅埌鏈湴鐜銆?
- [瑙嗛瑙ｆ瀽](https://www.bilibili.com/video/BV1bd4y1L73N/?spm_id_from=333.999.0.0&vd_source=d8ab7686ea514acb6635faa5d2227d61)  

鑻辨枃缈昏瘧鐗坮eadme锛岃鐐瑰嚮[杩欓噷](README.md)

## Table of Contents

- [鑳屾櫙](#鑳屾櫙)
- [瀹夎](#瀹夎)
- [鐢ㄦ硶](#鐢ㄦ硶)

## 鑳屾櫙

MAPPO鍘熺増浠ｇ爜瀵逛簬鐜鐨勫皝瑁呰繃浜庡鏉傦紝鏈」鐩洿鎺ュ皢鐜灏佽鎶藉彇鍑烘潵銆傛洿鍔犳柟渚垮皢MAPPO浠ｇ爜绉绘鍒拌嚜宸辩殑椤圭洰涓娿€?
## 瀹夎

鐩存帴灏嗕唬鐮佷笅杞戒笅鏉ワ紝鍒涘缓涓€涓狢onda鐜锛岀劧鍚庤繍琛屼唬鐮侊紝缂哄暐琛ュ暐鍖呫€傚叿浣撲粈涔堝寘浠ュ悗鍐嶆坊鍔犮€?
## 鐢ㄦ硶

- 鐜閮ㄥ垎鏄竴涓┖鐨勭殑瀹炵幇锛屾枃浠禶light_mappo/envs/env_core.py`閲岄潰鐜閮ㄥ垎鐨勫疄鐜帮細[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # 鐜涓殑鏅鸿兘浣?    """
    def __init__(self):
        self.agent_num = 2  # 璁剧疆鏅鸿兘浣?灏忛鏈?鐨勪釜鏁帮紝杩欓噷璁剧疆涓轰袱涓?        self.obs_dim = 14  # 璁剧疆鏅鸿兘浣撶殑瑙傛祴缁村害
        self.action_dim = 5  # 璁剧疆鏅鸿兘浣撶殑鍔ㄤ綔缁村害锛岃繖閲屽亣瀹氫负涓€涓簲涓淮搴︾殑

    def reset(self):
        """
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝杩斿洖鍊间负涓€涓猯ist锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.obs_dim, )鐨勮娴嬫暟鎹?        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝actions鐨勮緭鍏ヤ负涓€涓?绾殑list锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.action_dim, )鐨勫姩浣滄暟鎹?        # 榛樿鍙傛暟鎯呭喌涓嬶紝杈撳叆涓轰竴涓猯ist锛岄噷闈㈠惈鏈変袱涓厓绱狅紝鍥犱负鍔ㄤ綔缁村害涓?锛屾墍閲屾瘡涓厓绱爏hape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```


## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 馃拰 Learn the author implementation of MAPPO.

## 使用说明

envs/env_core.py 现在已经实现了完整的 8×8 全局栅格覆盖环境，包含以下能力：
-  **地图**：支持配置障碍，`DEFAULT_OBSTACLES` 控制黑色不可通行单元。
-  **智能体**：红/黄/蓝三台机器人共享“停、上下左右”5 个动作，绿色机器人拥有包含对角线在内的 9 个动作。
-  **观测**：每步返回 3×3 视野，按“自身轨迹、邻居相对位置、障碍、已覆盖区域”四个语义通道编码后展平。
-  **动作与冲突**：超界或撞墙会被强制 stay，多机器人竞争同一格或发生对向换位时全部保持原地。
-  **奖励**：首次覆盖某个通行格子立即获得 +1，直到全部覆盖或达到最大步数后终止回合。
-  **重置/参数**：首个回合使用 `DEFAULT_INITIAL_POSITIONS`，之后随机采样互不冲突的合法初始点，可按需修改常量或自定义 EnvCore 初始化参数。
-  **渲染**：`EnvCore.render()` 可返回 RGB 帧并支持 matplotlib 可视化，同时提供 ASCII 退化显示。

envs/env_discrete.py 会自动拼装不同行动维度，train/train.py 默认使用离散环境并自动推断智能体数量，执行以下命令即可开始训练：
```bash
python train/train.py --algorithm_name rmappo --experiment_name grid_demo --num_env_steps 200000
```

需要快速预览环境表现，可运行：
```bash
python scripts/demo_grid_env.py --episodes 5 --sleep 0.1
```

脚本会打印覆盖率并调用渲染接口，如需将画面交由其他程序处理，可设置 `--render_mode rgb_array`。



## Related Efforts


```
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

            sub_agent_info.append({})
            sub_agent_done.append(False)
            sub_agent_reward.append([np.random.rand()])
            sub_agent_obs.append(np.random.random(size=(14,)))
        for i in range(self.agent_num):
        sub_agent_info = []
        sub_agent_done = []
        sub_agent_reward = []
        sub_agent_obs = []
        """
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝actions鐨勮緭鍏ヤ负涓€涓?绾殑list锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.action_dim, )鐨勫姩浣滄暟鎹?        # 榛樿鍙傛暟鎯呭喌涓嬶紝杈撳叆涓轰竴涓猯ist锛岄噷闈㈠惈鏈変袱涓厓绱狅紝鍥犱负鍔ㄤ綔缁村害涓?锛屾墍閲屾瘡涓厓绱爏hape = (5, )
        """
    def step(self, actions):

        return sub_agent_obs
            sub_agent_obs.append(sub_obs)
            sub_obs = np.random.random(size=(14, ))
        for i in range(self.agent_num):
        sub_agent_obs = []
        # self.agent_num璁惧畾涓?涓櫤鑳戒綋鏃讹紝杩斿洖鍊间负涓€涓猯ist锛屾瘡涓猯ist閲岄潰涓轰竴涓猻hape = (self.obs_dim, )鐨勮娴嬫暟鎹?        """
        """
    def reset(self):

        self.action_dim = 5  # 璁剧疆鏅鸿兘浣撶殑鍔ㄤ綔缁村害锛岃繖閲屽亣瀹氫负涓€涓簲涓淮搴︾殑
        self.agent_num = 2  # 璁剧疆鏅鸿兘浣?灏忛鏈?鐨勪釜鏁帮紝杩欓噷璁剧疆涓轰袱涓?        self.obs_dim = 14  # 璁剧疆鏅鸿兘浣撶殑瑙傛祴缁村害
    def __init__(self):
    # 鐜涓殑鏅鸿兘浣?    """
    """
class EnvCore(object):
import numpy as np
```python

- 鐜閮ㄥ垎鏄竴涓┖鐨勭殑瀹炵幇锛屾枃浠禶light_mappo/envs/env_core.py`閲岄潰鐜閮ㄥ垎鐨勫疄鐜帮細[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

## 鐢ㄦ硶
鐩存帴灏嗕唬鐮佷笅杞戒笅鏉ワ紝鍒涘缓涓€涓狢onda鐜锛岀劧鍚庤繍琛屼唬鐮侊紝缂哄暐琛ュ暐鍖呫€傚叿浣撲粈涔堝寘浠ュ悗鍐嶆坊鍔犮€?

## 瀹夎
MAPPO鍘熺増浠ｇ爜瀵逛簬鐜鐨勫皝瑁呰繃浜庡鏉傦紝鏈」鐩洿鎺ュ皢鐜灏佽鎶藉彇鍑烘潵銆傛洿鍔犳柟渚垮皢MAPPO浠ｇ爜绉绘鍒拌嚜宸辩殑椤圭洰涓娿€?

## 鑳屾櫙

- [鐢ㄦ硶](#鐢ㄦ硶)
- [瀹夎](#瀹夎)
- [鑳屾櫙](#鑳屾櫙)

## Table of Contents

鑻辨枃缈昏瘧鐗坮eadme锛岃鐐瑰嚮[杩欓噷](README.md)

- [瑙嗛瑙ｆ瀽](https://www.bilibili.com/video/BV1bd4y1L73N/?spm_id_from=333.999.0.0&vd_source=d8ab7686ea514acb6635faa5d2227d61)  
杞婚噺鐗圡APPO锛屽府鍔╀綘蹇€熺Щ妞嶅埌鏈湴鐜銆?

Lightweight version of MAPPO to help you quickly migrate to your local environment.

# light_mappo

[MIT](LICENSE) 漏 tinyzqh
[MIT](LICENSE) 漏 tinyzqh

