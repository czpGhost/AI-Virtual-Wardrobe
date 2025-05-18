Page({
  data: {
    personSrc: '',
    garmentSrc: '',
    resultSrc: '',
    isLoading: false,
    errorMsg: ''
  },

  chooseImage(e) {
    const { type, source } = e.currentTarget.dataset; // 'person' or 'garment'; 'camera' or 'album'
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'], // 可以是 'original' 或 'compressed', 'compressed' 有助于减小体积
      sourceType: [source],
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        if (type === 'person') {
          this.setData({ personSrc: tempFilePath, errorMsg: '' });
        } else if (type === 'garment') {
          this.setData({ garmentSrc: tempFilePath, errorMsg: '' });
        }
      },
      fail: (err) => {
        console.error("选择图片失败:", err);
        if (err.errMsg !== "chooseImage:fail cancel") {
          this.setData({ errorMsg: '选择图片失败，请重试' });
        }
      }
    });
  },

  startGeneration() {
    if (!this.data.personSrc || !this.data.garmentSrc) {
      this.setData({ errorMsg: '请先上传人物和服装图片' });
      wx.showToast({
        title: '请先上传图片',
        icon: 'none'
      });
      return;
    }

    this.setData({ isLoading: true, resultSrc: '', errorMsg: '' });

    // 使用JSON+Base64方式发送请求
    const apiUrl = 'https://5fqtlb6nrk8-ghost.gear-c1.openbayes.net/virtual-tryon';
    const fileSystemManager = wx.getFileSystemManager();
    
    // 读取两张图片的Base64数据
    Promise.all([
      new Promise((resolve, reject) => {
        fileSystemManager.readFile({
          filePath: this.data.personSrc,
          encoding: 'base64',
          success: res => resolve(res.data),
          fail: reject
        });
      }),
      new Promise((resolve, reject) => {
        fileSystemManager.readFile({
          filePath: this.data.garmentSrc,
          encoding: 'base64',
          success: res => resolve(res.data),
          fail: reject
        });
      })
    ]).then(([personBase64, garmentBase64]) => {
      // 尝试使用FormData方式发送请求，适应服务器端的Form参数期望
      const formData = {
        person_image_base64: personBase64,
        garment_image_base64: garmentBase64,
        model_type: 'dress_code',
        garment_type: 'dresses',
        preprocess_garment: 'true',
        acceleration: 'true',
        steps: '88',
        guidance_scale: '3.0',
        seed: '36',
        repaint_mode: 'true'
      };

      // 使用wx.request发送FormData
      wx.request({
        url: apiUrl,
        method: 'POST',
        data: formData,
        header: {
          'content-type': 'application/x-www-form-urlencoded'  // 改为表单内容类型
        },
        success: (res) => {
          if (res.statusCode === 200 && res.data && res.data.image_base64) {
            this.setData({
              resultSrc: 'data:image/png;base64,' + res.data.image_base64,
              isLoading: false
            });
            wx.showToast({ title: '生成成功！', icon: 'success' });
          } else {
            this.handleApiError(res);
          }
        },
        fail: (err) => {
          console.error("API请求失败:", err);
          this.setData({
            errorMsg: '请求后端服务失败，请检查网络或联系管理员。',
            isLoading: false
          });
        }
      });
    }).catch(err => {
      console.error("读取图片文件Base64失败:", err);
      this.setData({
        errorMsg: '处理图片文件失败，请重试。',
        isLoading: false
      });
    });
  },
  
  // 通用错误处理函数
  handleApiError(res) {
    console.error(`API错误 - Status: ${res.statusCode}, Data:`, res.data);
    let errorMsg = `生成失败: 服务器错误 (状态码: ${res.statusCode})`;
    
    if (res.data) {
      if (res.data.detail) {
        // 确保我们能在控制台中看到完整的详细信息
        console.error("错误详情:", res.data.detail);
        
        // 尝试格式化detail的内容以便更好地显示
        if (Array.isArray(res.data.detail)) {
          // 详细记录每个错误项
          res.data.detail.forEach((item, index) => {
            console.error(`错误项 ${index + 1}:`, item);
            if (item.loc) console.error(`- 位置:`, item.loc);
            if (item.msg) console.error(`- 消息:`, item.msg);
            if (item.type) console.error(`- 类型:`, item.type);
          });
          
          // 创建用户友好的错误信息
          const detailStr = res.data.detail
            .map(d => `[${d.type || '错误'}: ${d.msg || '未知'} at ${d.loc ? d.loc.join('.') : '未知位置'}]`)
            .join('; ');
          errorMsg = `生成失败: ${detailStr} (状态码: ${res.statusCode})`;
        } else {
          // 如果detail不是数组，直接使用它的字符串表示
          errorMsg = `生成失败: ${JSON.stringify(res.data.detail)} (状态码: ${res.statusCode})`;
        }
      } else if (typeof res.data === 'string') {
        errorMsg = `生成失败: ${res.data} (状态码: ${res.statusCode})`;
      }
    }
    
    this.setData({
      errorMsg: errorMsg,
      isLoading: false
    });
  },

  // 预览图片
  previewImage: function(e) {
    const src = e.currentTarget.dataset.src;
    wx.previewImage({
      current: src,
      urls: [src]
    });
  },

  // 保存图片到相册
  saveImage: function() {
    if (!this.data.resultSrc) return;
    
    wx.getSetting({
      success: (res) => {
        if (!res.authSetting['scope.writePhotosAlbum']) {
          wx.authorize({
            scope: 'scope.writePhotosAlbum',
            success: () => {
              this.saveImageToAlbum();
            },
            fail: () => {
              wx.showModal({
                title: '提示',
                content: '需要您授权保存图片到相册',
                showCancel: false
              });
            }
          });
        } else {
          this.saveImageToAlbum();
        }
      }
    });
  },

  // 实际保存图片的函数
  saveImageToAlbum: function() {
    wx.showLoading({
      title: '保存中...',
    });
    
    wx.downloadFile({
      url: this.data.resultSrc,
      success: (res) => {
        if (res.statusCode === 200) {
          wx.saveImageToPhotosAlbum({
            filePath: res.tempFilePath,
            success: () => {
              wx.showToast({
                title: '保存成功',
                icon: 'success'
              });
            },
            fail: () => {
              wx.showToast({
                title: '保存失败',
                icon: 'none'
              });
            },
            complete: () => {
              wx.hideLoading();
            }
          });
        } else {
          wx.hideLoading();
          wx.showToast({
            title: '下载失败',
            icon: 'none'
          });
        }
      },
      fail: () => {
        wx.hideLoading();
        wx.showToast({
          title: '下载失败',
          icon: 'none'
        });
      }
    });
  },

  // 分享图片
  shareImage: function() {
    // 基本实现，在真实环境中可能需要根据微信小程序的分享机制调整
    wx.showShareMenu({
      withShareTicket: true,
      menus: ['shareAppMessage', 'shareTimeline']
    });
  },

  // 关闭错误提示
  closeError: function() {
    this.setData({ errorMsg: '' });
  }
}) 