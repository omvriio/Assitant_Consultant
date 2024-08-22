if (chrome.sidePanel) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
} else {
  chrome.action.onClicked.addListener(() => {
    chrome.notifications.create({
      type: "basic",
      title: "Unsupported",
      iconUrl: chrome.runtime.getURL("icon.png"),
      message: "Please upgrade your Chrome browser to version 114+",
    });
  });
}

// Remove the declarativeNetRequest.updateDynamicRules part as it's not needed anymore