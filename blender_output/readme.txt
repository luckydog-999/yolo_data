将code_in_blender置于blender脚本中进行运行，在运行脚本前确认一下，是否进行了下述操作：

1.物体已经进行了贴图还有材质渲染，针对不同物体进行不同的渲染方式，物体材质可以通过材质包下载

2.环境是否选择为模型应用的场景，环境下载可以去https://polyhaven.com/zh/a/peppermint_powerplant
具体操作如下：
下载 HDRI (如果你还没有)：
去 Poly Haven 下载一张 Indoor (室内) 或 Studio (摄影棚)(根据你的场景选择) 的 HDRI 图片（下载 1k 或 2k 分辨率的 .exr 或 .hdr 格式即可，太大了加载慢）。
在 Blender 中加载：
在右侧属性面板，点击 World Properties (世界属性)（红色地球图标）。
找到 Color (颜色)，点击它旁边的小黄点。
在弹出的菜单中选择 Environment Texture (环境纹理)。（注意：千万别选成 Image Texture，那是给物体贴图用的）。
点击 Open，选择你刚下载的 HDRI 文件。

3.进行运行脚本code_in_blender.py前的准备(脚本在blender里面运行)
具体操作如下：
第一步让背景透明 (Film Transparent)
为了方便后期合成任意背景（增加数据多样性），通常我们只渲染物体本身，背景设为透明。
点击右侧面板的 Render Properties (渲染属性)（白色照相机图标）。
找到 Film (胶片) 选项卡，展开它。
勾选 Transparent (透明)。
此时你会看到 HDRI 的光照效果还在，但背景变成了像 PS 一样的灰白格子，这是对的。

第二部设置物体 ID (Pass Index)
我们需要给你的物体一个“身份证号”，这样脚本才能在渲染时生成纯黑白的掩码图（Mask），YOLO 需要这个来计算分割轮廓。
选中你的物体。
点击右侧面板的 Object Properties (物体属性)（橙色方框图标）。
找到 Relations (关系) 选项卡。
将 Pass Index (通道索引) 设置为 1。
这就告诉 Blender：这个物体是“1号目标”。

第三步开启输出通道
点击右侧面板的 View Layer Properties (视图层属性)（像一摞照片的图标）。
向下滚动找到 Data (数据) -> Passes (通道)（在某些版本里直接叫 Passes）。
勾选 Object Index (物体索引)。

注意可以先通过测试代码test_speed.py看下运行速度，如果卡死大概率没有开启gpu，去blender编辑 - 偏好设置 - 系统 选择GPU进行加速
并且通过打印机的图标降低图片分辨率，并且减少渲染的采样到256，从而达到加速

4.将code_in_blender置于blender脚本中进行运行。

5.制作出来的数据集因为背景是透明的，所以可以制作一个background文件夹，对其任意添加背景

*******************************************************************************************
其实不勾选背景透明也行，可以后续换背景继续生成更方便！！！！！！！！！！！！！