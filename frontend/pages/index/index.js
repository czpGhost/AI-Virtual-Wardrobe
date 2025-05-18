Page({
    data: {},
    navigateToTryonUpper() {
      wx.navigateTo({url: '/pages/tryon_upper/tryon_upper'});
    },
    navigateToTryonLower() {
      wx.navigateTo({url: '/pages/tryon_lower/tryon_lower'});
    },
    navigateToTryonDress() {
      wx.navigateTo({url: '/pages/tryon_dress/tryon_dress'});
    },
    navigateToTransfer() {
      wx.navigateTo({url: '/pages/transfer/transfer'});
    }
  });