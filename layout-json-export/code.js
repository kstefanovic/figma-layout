figma.showUI(__html__);

const humanizeType = (type) => type.toLowerCase().replace(/_/g, " ");

const getOrigin = (node) => {
  const t = node.absoluteTransform;
  return { x: t[0][2], y: t[1][2] };
};

const absoluteBox = (node, origin) => {
  const t = node.absoluteTransform;
  return {
    x: Number((t[0][2] - origin.x).toFixed(2)),
    y: Number((t[1][2] - origin.y).toFixed(2)),
    width: Number(node.width.toFixed(2)),
    height: Number(node.height.toFixed(2)),
  };
};

const serializeNode = (node, origin) => {
  const base = {
    id: node.id,
    name: node.name,
    type: humanizeType(node.type),
    bounds: absoluteBox(node, origin),
    visible: node.visible,
    opacity:
      typeof node.opacity === "number" ? Number(node.opacity.toFixed(2)) : 1,
  };

  if ("characters" in node) {
    base.characters = node.characters;
  }

  if ("layoutMode" in node) {
    base.layoutMode = node.layoutMode;
    base.itemSpacing = node.itemSpacing;
    base.padding = {
      top: node.paddingTop,
      right: node.paddingRight,
      bottom: node.paddingBottom,
      left: node.paddingLeft,
    };
  }

  if ("children" in node && Array.isArray(node.children)) {
    base.children = node.children.map((child) => serializeNode(child, origin));
  }

  return base;
};

const collectElements = (node, origin, bucket) => {
  bucket.push({
    id: node.id,
    name: node.name,
    type: humanizeType(node.type),
    bounds: absoluteBox(node, origin),
  });
  if ("children" in node && Array.isArray(node.children)) {
    for (const child of node.children) {
      collectElements(child, origin, bucket);
    }
  }
};

const findTargetFrame = () => {
  const selection = figma.currentPage.selection;
  const candidate = selection.find((node) => "children" in node);
  if (candidate) return candidate;

  // If user selected a leaf node, climb to its nearest parent that has children.
  const first = selection[0];
  let parent = first && "parent" in first ? first.parent : null;
  while (parent) {
    if ("children" in parent) return parent;
    parent = "parent" in parent ? parent.parent : null;
  }

  return null;
};

const findNodeById = (root, id) => {
  if (root.id === id) return root;
  if ("children" in root && Array.isArray(root.children)) {
    for (const child of root.children) {
      const found = findNodeById(child, id);
      if (found) return found;
    }
  }
  return null;
};

const toNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? Number(n.toFixed(2)) : null;
};

const applyBoundsOverrides = (bounds, overrides) => {
  const next = Object.assign({}, bounds);
  const x = toNumber(overrides.x);
  const y = toNumber(overrides.y);
  const width = toNumber(overrides.width);
  const height = toNumber(overrides.height);

  if (x !== null) next.x = x;
  if (y !== null) next.y = y;
  if (width !== null) next.width = width;
  if (height !== null) next.height = height;

  return next;
};

figma.ui.onmessage = (msg) => {
  if (msg.type === "get-top-level-frames") {
    const topLevelFrames = figma.currentPage.children
      .filter((node) => node.type === "FRAME")
      .map((frameNode) => {
        const origin = getOrigin(frameNode);
        const payload = serializeNode(frameNode, origin);
        if (!payload.children || payload.children.length !== 8) {
          return null;
        }
        return Object.assign({}, payload, { templateId });
      })
      .filter((node) => !!node);
    return;
  }

  if (msg.type === "export-selected-frames") {
    const selection = figma.currentPage.selection;
    const templateId = msg.templateId;
    if (!selection.length) {
      figma.notify("No items selected");
      return;
    }
    if (!templateId) {
      figma.ui.postMessage({
        type: "error",
        msg: "Input template id",
      });
      return;
    }

    const layoutData = selection
      .map((frameNode) => {
        const origin = getOrigin(frameNode);
        const payload = serializeNode(frameNode, origin);
        if (!payload.children) {
          return null;
        }
        return Object.assign({}, payload, { templateId });
      })
      .filter((node) => !!node);

    figma.ui.postMessage({
      type: "export-json",
      templateId,
      data: layoutData,
    });
    return;
  }

  if (!msg || msg.type !== "override-element") return;
};
