from modelscope.hub.snapshot_download import snapshot_download

# 下载官方匹配的DDColor模型到本地modelscope文件夹
model_dir = snapshot_download('damo/cv_ddcolor_image-colorization', cache_dir='./modelscope')
print('✅ 官方模型下载完成，文件保存到:', model_dir)