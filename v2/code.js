/**
 * Figma plugin main thread. Flows:
 * - ``POST …/pipeline/banner-raw-to-target-json-json`` — banner + raw JSON + target size → layout clone.
 * - ``POST …/figma/convert-semantic-json`` — banner + element grid PNG + raw JSON (atlas regions) → semantic JSON via Qwen.
 * - HTML/CSS export from serialized JSON + assets (local).
 */
figma.showUI(__html__, { width: 400, height: 720 });

function normalizeType(type) {
  return String(type || "").toLowerCase().replace(/_/g, " ");
}

function getOrigin(node) {
  const t = node.absoluteTransform;
  return { x: t[0][2], y: t[1][2] };
}

function absoluteBox(node, origin) {
  const t = node.absoluteTransform;
  return {
    x: Number((t[0][2] - origin.x).toFixed(2)),
    y: Number((t[1][2] - origin.y).toFixed(2)),
    width: Number(node.width.toFixed(2)),
    height: Number(node.height.toFixed(2))
  };
}

function serializeNode(node, origin, path) {
  const base = {
    id: node.id,
    path: path,
    name: node.name,
    type: normalizeType(node.type),
    bounds: absoluteBox(node, origin),
    visible: node.visible !== false,
    opacity: typeof node.opacity === "number" ? Number(node.opacity.toFixed(3)) : 1
  };

  if ("characters" in node) {
    base.characters = node.characters;
    if ("fontSize" in node) base.fontSize = node.fontSize;
    if ("fontName" in node) base.fontName = node.fontName;
    if ("textAlignHorizontal" in node) base.textAlignHorizontal = node.textAlignHorizontal;
    if ("textAlignVertical" in node) base.textAlignVertical = node.textAlignVertical;
  }

  if ("layoutMode" in node) {
    base.layoutMode = node.layoutMode;
    base.itemSpacing = node.itemSpacing;
    base.padding = {
      top: node.paddingTop,
      right: node.paddingRight,
      bottom: node.paddingBottom,
      left: node.paddingLeft
    };
  }

  if ("children" in node && Array.isArray(node.children)) {
    base.children = node.children.map((child, index) => {
      const childPath = path === "" ? String(index) : `${path}/${index}`;
      return serializeNode(child, origin, childPath);
    });
  }

  return base;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function cssIdent(value, fallback) {
  const raw = String(value || fallback || "node")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return raw || fallback || "node";
}

function cssNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Number(n.toFixed(2)) : fallback;
}

function htmlCssNodeLines(node, rootBounds, indexPath, depth) {
  if (!node || typeof node !== "object") return [];
  const bounds = node.bounds && typeof node.bounds === "object" ? node.bounds : {};
  const type = normalizeType(node.type || "");
  const isText = type === "text";
  const name = node.name || type || "node";
  const className = `node-${cssIdent(indexPath || "root", "root")}`;
  const x = cssNumber(bounds.x, 0);
  const y = cssNumber(bounds.y, 0);
  const w = Math.max(1, cssNumber(bounds.width, 1));
  const h = Math.max(1, cssNumber(bounds.height, 1));
  const styleParts = [
    `left:${x}px`,
    `top:${y}px`,
    `width:${w}px`,
    `height:${h}px`,
    `opacity:${typeof node.opacity === "number" ? node.opacity : 1}`,
  ];
  if (isText) {
    const fs = typeof node.fontSize === "number" ? Math.max(1, cssNumber(node.fontSize, 16)) : Math.max(10, Math.min(48, h * 0.42));
    styleParts.push(`font-size:${fs}px`, "font-weight:700", "line-height:1.05", "color:#111827", "white-space:pre-wrap");
  } else {
    styleParts.push("background:rgba(148,163,184,0.22)", "border:1px solid rgba(15,23,42,0.16)");
  }
  const indent = "  ".repeat(depth);
  const tag = isText ? "div" : "div";
  const label = isText ? node.characters || "" : "";
  const children = Array.isArray(node.children) ? node.children : [];
  const dataAttrs = `data-figma-id="${escapeHtml(node.id || "")}" data-name="${escapeHtml(name)}" data-type="${escapeHtml(type)}"`;
  const lines = [
    `${indent}<${tag} class="figma-node ${isText ? "figma-text" : "figma-shape"} ${className}" ${dataAttrs} style="${styleParts.join(";")}">`,
  ];
  if (isText) {
    lines.push(`${indent}  ${escapeHtml(label)}`);
  }
  for (let i = 0; i < children.length; i++) {
    const childPath = indexPath ? `${indexPath}-${i}` : String(i);
    lines.push(...htmlCssNodeLines(children[i], rootBounds, childPath, depth + 1));
  }
  lines.push(`${indent}</${tag}>`);
  return lines;
}

function rawJsonToHtmlCss(rawJson, bannerPngBase64, elementAssets) {
  const bounds = rawJson && rawJson.bounds && typeof rawJson.bounds === "object" ? rawJson.bounds : {};
  const width = Math.max(1, cssNumber(bounds.width, 1));
  const height = Math.max(1, cssNumber(bounds.height, 1));
  const title = escapeHtml(rawJson.name || "Figma export");
  const children = Array.isArray(rawJson.children) ? rawJson.children : [];
  const childLines = [];
  const assets = Array.isArray(elementAssets) ? elementAssets : [];
  if (bannerPngBase64) {
    childLines.push(
      `    <img class="figma-render" alt="${title}" src="data:image/png;base64,${bannerPngBase64}" />`,
    );
  }
  if (assets.length > 0) {
    for (const asset of assets) {
      const b = asset.bounds || {};
      const x = cssNumber(b.x, 0);
      const y = cssNumber(b.y, 0);
      const w = Math.max(1, cssNumber(b.width, 1));
      const h = Math.max(1, cssNumber(b.height, 1));
      childLines.push(
        `    <img class="figma-node figma-asset" data-path="${escapeHtml(asset.path || "")}" data-figma-id="${escapeHtml(asset.id || "")}" data-name="${escapeHtml(asset.name || "")}" data-type="${escapeHtml(asset.type || "")}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;opacity:${asset.opacity == null ? 1 : asset.opacity}" src="data:image/png;base64,${asset.pngBase64}" alt="${escapeHtml(asset.name || asset.path || "figma element")}" />`,
      );
    }
  } else if (!bannerPngBase64) {
    for (let i = 0; i < children.length; i++) {
      childLines.push(...htmlCssNodeLines(children[i], bounds, String(i), 2));
    }
  }
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title}</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #0f172a;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .figma-banner {
      position: relative;
      width: ${width}px;
      height: ${height}px;
      overflow: hidden;
      background: #fff;
    }
    .figma-node {
      position: absolute;
      overflow: hidden;
    }
    .figma-text {
      background: transparent;
      border: 0;
      display: flex;
      align-items: center;
    }
    .figma-render {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: fill;
      display: block;
      z-index: 0;
    }
    .figma-asset {
      display: block;
      object-fit: fill;
      pointer-events: auto;
      opacity: 0 !important;
      z-index: 1;
    }
    .figma-asset:hover {
      opacity: 0.18 !important;
      outline: 1px dashed rgba(255,255,255,0.9);
    }
  </style>
</head>
<body>
  <main class="figma-banner" data-figma-id="${escapeHtml(rawJson.id || "")}" data-name="${title}">
${childLines.join("\n")}
  </main>
</body>
</html>`;
}

async function exportFramePngBytes(node) {
  return await node.exportAsync({
    format: "PNG",
    constraint: { type: "SCALE", value: 1 }
  });
}

async function exportElementAssetsForHtml(root, rawJson, maxCount) {
  const origin = getOrigin(root);
  const entries = collectLeafElementRefs(root, maxCount);
  const assets = [];
  for (const { path, node } of entries) {
    try {
      const bytes = await exportFramePngBytes(node);
      assets.push({
        path,
        id: node.id,
        name: node.name,
        type: normalizeType(node.type),
        bounds: absoluteBox(node, origin),
        opacity: typeof node.opacity === "number" ? Number(node.opacity.toFixed(3)) : 1,
        pngBase64: uint8ToBase64(bytes),
      });
    } catch (e) {
      console.warn("HTML export: failed exporting element", path, node && node.name, e);
    }
  }
  return assets;
}

/** Max leaves packed into the element atlas. Keep in sync with backend MAX_ATLAS_REGIONS. */
const MAX_ELEMENT_LAYER_PNGS = 512;

/** Space between element bounding boxes. Requested as 20x the previous 12px spacing. */
const PREVIOUS_ATLAS_GAP = 12;
const ATLAS_GAP = PREVIOUS_ATLAS_GAP * 20;
const ATLAS_CELL_PADDING = Math.round(ATLAS_GAP / 2);
const ATLAS_MAX_ROW_WIDTH = 8192;
const ATLAS_MAX_CELL = 4096;
const ELEMENTS_PNG_MAX_WIDTH = 1920;
const ELEMENTS_PNG_MAX_HEIGHT = 1028;
const ATLAS_BOX_COLOR = { r: 1, g: 0, b: 0 };
const ATLAS_CELL_COLOR = { r: 1, g: 1, b: 1 };

/**
 * Collect **leaf** scene nodes (same rules as former per-PNG export): no children
 * (except INSTANCE as a single leaf), visible, min size 1px.
 */
function collectLeafElementRefs(root, maxCount) {
  const out = [];
  let count = 0;

  function visit(node, path) {
    if (count >= maxCount) return;
    if (node.visible === false) return;
    if (!("width" in node) || node.width < 1 || node.height < 1) return;

    const kids =
      "children" in node && Array.isArray(node.children) ? node.children : [];
    const hasKids = kids.length > 0;

    if (hasKids && node.type !== "INSTANCE") {
      for (let i = 0; i < kids.length; i++) {
        const childPath = path === "" ? String(i) : `${path}/${i}`;
        visit(kids[i], childPath);
        if (count >= maxCount) return;
      }
      return;
    }

    out.push({ path, node });
    count++;
  }

  const top = root.children;
  if (!top || !top.length) return out;
  for (let i = 0; i < top.length; i++) {
    visit(top[i], String(i));
    if (count >= maxCount) break;
  }
  if (count >= maxCount) {
    console.warn("collectLeafElementRefs: hit backend atlas-region cap", maxCount);
  }
  return out;
}

function atlasExportScale(width, height) {
  const w = Math.max(1, Number(width) || 1);
  const h = Math.max(1, Number(height) || 1);
  return Math.min(1, ELEMENTS_PNG_MAX_WIDTH / w, ELEMENTS_PNG_MAX_HEIGHT / h);
}

function scaledRegionValue(value, scale) {
  return Math.max(0, Math.round((Number(value) || 0) * scale));
}

function scaledRegionSize(value, scale) {
  return Math.max(1, Math.round((Number(value) || 0) * scale));
}

function makeBoundingBoxRect(x, y, width, height, exportScale) {
  const rect = figma.createRectangle();
  rect.name = "__element_bbox__";
  rect.x = x;
  rect.y = y;
  rect.resizeWithoutConstraints(Math.max(1, width), Math.max(1, height));
  rect.fills = [];
  rect.strokes = [{ type: "SOLID", color: ATLAS_BOX_COLOR }];
  rect.strokeWeight = 1;
  rect.strokeAlign = "INSIDE";
  return rect;
}

function makeAtlasCellFrame(x, y, width, height) {
  const cell = figma.createFrame();
  cell.name = "__element_cell__";
  cell.x = x;
  cell.y = y;
  cell.layoutMode = "NONE";
  cell.clipsContent = true;
  cell.fills = [
    {
      type: "SOLID",
      color: ATLAS_CELL_COLOR,
      opacity: 0.04,
    },
  ];
  cell.resizeWithoutConstraints(Math.max(1, width), Math.max(1, height));
  return cell;
}

/** Load a legible UI font for atlas id labels (Figma availability varies). */
async function loadAtlasLabelFont() {
  const candidates = [
    { family: "Inter", style: "Bold" },
    { family: "Inter", style: "Regular" },
    { family: "Roboto", style: "Bold" },
    { family: "Roboto", style: "Regular" },
  ];
  for (const f of candidates) {
    try {
      await figma.loadFontAsync(f);
      return f;
    } catch (_e) {
      /* try next */
    }
  }
  return null;
}

/**
 * Clone leaves into one off-screen frame, pack in rows with large spacing, draw visible
 * bounding boxes, and export a **single** PNG atlas capped to 1920 x 1028.
 * Returns PNG bytes + Base64 + region list in final exported pixel coords. Names/paths match ``raw_json``.
 */
async function buildElementAtlasPngAndRegions(root, maxCount) {
  const entries = collectLeafElementRefs(root, maxCount);
  if (!entries.length) {
    return {
      atlasPngBase64: "",
      atlasPngBytes: new Uint8Array(0),
      regions: [],
      atlasSize: { width: 0, height: 0, source_width: 0, source_height: 0, scale: 1 },
    };
  }

  const atlas = figma.createFrame();
  atlas.name = "__plugin_element_atlas__";
  atlas.fills = [];
  atlas.layoutMode = "NONE";
  atlas.clipsContent = false;
  figma.currentPage.appendChild(atlas);
  atlas.x = -120000;
  atlas.y = -120000;

  const layoutRegions = [];
  const bboxRects = [];
  let curX = 0;
  let curY = 0;
  let rowH = 0;

  try {
    const labelFont = await loadAtlasLabelFont();
    const headerH = labelFont ? 46 : 0;

    for (const { path, node } of entries) {
      let clone;
      try {
        clone = node.clone();
      } catch (e) {
        console.warn("buildElementAtlas: clone failed", path, e);
        continue;
      }

      try {
        if ("resizeWithoutConstraints" in clone && typeof clone.resizeWithoutConstraints === "function") {
          const tw = Math.min(clone.width, ATLAS_MAX_CELL);
          const th = Math.min(clone.height, ATLAS_MAX_CELL);
          if (tw < clone.width || th < clone.height) {
            clone.resizeWithoutConstraints(tw, th);
          }
        }
      } catch (e) {
        /* keep natural size */
      }

      const cw = clone.width;
      const ch = clone.height;
      const cellW = cw + ATLAS_CELL_PADDING * 2;
      const cellH = headerH + ATLAS_CELL_PADDING + ch + ATLAS_CELL_PADDING;

      if (curX + cellW + ATLAS_GAP > ATLAS_MAX_ROW_WIDTH && curX > 0) {
        curY += rowH + ATLAS_GAP;
        curX = 0;
        rowH = 0;
      }

      const cell = makeAtlasCellFrame(curX, curY, cellW, cellH);
      atlas.appendChild(cell);

      if (labelFont) {
        const hdr = figma.createRectangle();
        hdr.name = "__atlas_cell_header_bg__";
        hdr.resize(cellW, headerH);
        hdr.x = 0;
        hdr.y = 0;
        hdr.fills = [{ type: "SOLID", color: { r: 0.92, g: 0.93, b: 0.95 } }];
        hdr.strokes = [{ type: "SOLID", color: { r: 0.72, g: 0.76, b: 0.82 } }];
        hdr.strokeWeight = 1;
        hdr.strokeAlign = "INSIDE";
        cell.appendChild(hdr);
        try {
          const idLabel = figma.createText();
          idLabel.name = "__atlas_id_label__";
          idLabel.fontName = labelFont;
          idLabel.fontSize = Math.min(20, Math.max(14, Math.round(headerH * 0.48)));
          const rawId = String(node.id || path || "");
          const idText = rawId.length > 40 ? rawId.slice(0, 37) + "…" : rawId;
          idLabel.characters = `id:${idText}`;
          idLabel.fills = [{ type: "SOLID", color: { r: 0.05, g: 0.08, b: 0.12 } }];
          cell.appendChild(idLabel);
          idLabel.x = 6;
          idLabel.y = Math.max(2, (headerH - idLabel.height) / 2);
        } catch (te) {
          console.warn("buildElementAtlas: id label failed", path, te);
        }
      }

      cell.appendChild(clone);
      clone.x = ATLAS_CELL_PADDING;
      clone.y = headerH + ATLAS_CELL_PADDING;
      const bboxRect = makeBoundingBoxRect(ATLAS_CELL_PADDING, headerH + ATLAS_CELL_PADDING, cw, ch, 1);
      cell.appendChild(bboxRect);
      bboxRects.push(bboxRect);

      layoutRegions.push({
        path,
        node_id: node.id,
        name: node.name,
        type: normalizeType(node.type),
        atlas_x: Math.round(curX + ATLAS_CELL_PADDING),
        atlas_y: Math.round(curY + headerH + ATLAS_CELL_PADDING),
        atlas_width: Math.round(cw),
        atlas_height: Math.round(ch),
        cell_x: Math.round(curX),
        cell_y: Math.round(curY),
        cell_width: Math.round(cellW),
        cell_height: Math.round(cellH),
      });

      curX += cellW + ATLAS_GAP;
      rowH = Math.max(rowH, cellH);
    }

    let maxR = 0;
    let maxB = 0;
    for (const region of layoutRegions) {
      maxR = Math.max(maxR, region.cell_x + region.cell_width);
      maxB = Math.max(maxB, region.cell_y + region.cell_height);
    }
    const finalW = Math.max(1, Math.ceil(maxR));
    const finalH = Math.max(1, Math.ceil(maxB));
    const scale = atlasExportScale(finalW, finalH);
    for (const bboxRect of bboxRects) {
      bboxRect.strokeWeight = Math.max(1, Math.min(2, 0.75 + scale));
    }

    if ("resizeWithoutConstraints" in atlas) {
      atlas.resizeWithoutConstraints(finalW, finalH);
    }

    const bytes = await atlas.exportAsync({
      format: "PNG",
      constraint: { type: "SCALE", value: scale },
    });
    const regions = layoutRegions.map((region) =>
      Object.assign({}, region, {
        atlas_x: scaledRegionValue(region.atlas_x, scale),
        atlas_y: scaledRegionValue(region.atlas_y, scale),
        atlas_width: scaledRegionSize(region.atlas_width, scale),
        atlas_height: scaledRegionSize(region.atlas_height, scale),
        atlas_cell_x: scaledRegionValue(region.cell_x, scale),
        atlas_cell_y: scaledRegionValue(region.cell_y, scale),
        atlas_cell_width: scaledRegionSize(region.cell_width, scale),
        atlas_cell_height: scaledRegionSize(region.cell_height, scale),
        atlas_scale: Number(scale.toFixed(6)),
      }),
    );
    return {
      atlasPngBytes: new Uint8Array(bytes),
      atlasPngBase64: uint8ToBase64(bytes),
      regions,
      atlasSize: {
        width: scaledRegionSize(finalW, scale),
        height: scaledRegionSize(finalH, scale),
        source_width: finalW,
        source_height: finalH,
        scale: Number(scale.toFixed(6)),
      },
    };
  } finally {
    atlas.remove();
  }
}

/**
 * Add ``atlas_region: { x, y, width, height }`` on each ``raw_json`` node whose ``path``
 * appears in the atlas (same ``path`` / ``name`` as serialization).
 */
function injectAtlasRegionsIntoRawJson(rawJson, regions) {
  if (!rawJson || !Array.isArray(regions) || regions.length === 0) return;
  const byPath = new Map(
    regions.map((r) => [
      r.path,
      {
        x: r.atlas_x,
        y: r.atlas_y,
        width: r.atlas_width,
        height: r.atlas_height,
      },
    ]),
  );

  function walk(n) {
    if (!n || typeof n !== "object") return;
    if (typeof n.path === "string" && byPath.has(n.path)) {
      n.atlas_region = byPath.get(n.path);
    }
    if (Array.isArray(n.children)) {
      n.children.forEach(walk);
    }
  }

  walk(rawJson);
}

function attachAtlasMetadataToRawJson(rawJson, atlasSize, regions) {
  if (!rawJson || typeof rawJson !== "object") return;
  rawJson.element_atlas = {
    file_name: "elements.png",
    max_width: ELEMENTS_PNG_MAX_WIDTH,
    max_height: ELEMENTS_PNG_MAX_HEIGHT,
    width: atlasSize && atlasSize.width ? atlasSize.width : 0,
    height: atlasSize && atlasSize.height ? atlasSize.height : 0,
    source_width: atlasSize && atlasSize.source_width ? atlasSize.source_width : 0,
    source_height: atlasSize && atlasSize.source_height ? atlasSize.source_height : 0,
    scale: atlasSize && atlasSize.scale ? atlasSize.scale : 1,
    region_count: Array.isArray(regions) ? regions.length : 0,
    bbox_gap_px: ATLAS_GAP,
    bbox_style:
      "each cell: grey header strip with id:<node_id> text, then element thumbnail; red stroke around thumbnail",
  };
}

function uint8ToBase64(bytes) {
  const base64abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let result = "";
  let i;

  for (i = 0; i + 2 < bytes.length; i += 3) {
    result += base64abc[bytes[i] >> 2];
    result += base64abc[((bytes[i] & 0x03) << 4) | (bytes[i + 1] >> 4)];
    result += base64abc[((bytes[i + 1] & 0x0f) << 2) | (bytes[i + 2] >> 6)];
    result += base64abc[bytes[i + 2] & 0x3f];
  }

  if (i < bytes.length) {
    result += base64abc[bytes[i] >> 2];

    if (i === bytes.length - 1) {
      result += base64abc[(bytes[i] & 0x03) << 4];
      result += "==";
    } else {
      result += base64abc[((bytes[i] & 0x03) << 4) | (bytes[i + 1] >> 4)];
      result += base64abc[(bytes[i + 1] & 0x0f) << 2];
      result += "=";
    }
  }

  return result;
}

function concatUint8Arrays(pieces) {
  let total = 0;
  for (const p of pieces) {
    total += p.length;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of pieces) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** UTF-8 encode string to bytes (Figma main thread has no FormData/Blob). */
function utf8Bytes(s) {
  const str = String(s);
  if (typeof TextEncoder !== "undefined") {
    return new TextEncoder().encode(str);
  }
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let c = str.charCodeAt(i);
    if (c < 0x80) {
      bytes.push(c);
    } else if (c < 0x800) {
      bytes.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
    } else if (c < 0xd800 || c >= 0xe000) {
      bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    } else {
      i++;
      c = 0x10000 + (((c & 0x3ff) << 10) | (str.charCodeAt(i) & 0x3ff));
      bytes.push(
        0xf0 | (c >> 18),
        0x80 | ((c >> 12) & 0x3f),
        0x80 | ((c >> 6) & 0x3f),
        0x80 | (c & 0x3f),
      );
    }
  }
  return new Uint8Array(bytes);
}

/**
 * multipart/form-data without FormData (not available in Figma plugin sandbox).
 * @param {string} boundary
 * @param {{ name: string, filename?: string | null, contentType: string, body: Uint8Array }}[] parts
 */
function buildMultipartFormDataBody(boundary, parts) {
  const chunks = [];
  for (const part of parts) {
    chunks.push(utf8Bytes("--" + boundary + "\r\n"));
    let head = 'Content-Disposition: form-data; name="' + part.name + '"';
    if (part.filename) {
      head += '; filename="' + part.filename + '"';
    }
    head += "\r\nContent-Type: " + part.contentType + "\r\n\r\n";
    chunks.push(utf8Bytes(head));
    const body = part.body instanceof Uint8Array ? part.body : new Uint8Array(part.body);
    chunks.push(body);
    chunks.push(utf8Bytes("\r\n"));
  }
  chunks.push(utf8Bytes("--" + boundary + "--\r\n"));
  return concatUint8Arrays(chunks);
}

function stampOriginalNodeIds(root) {
  let stamped = 0;

  function walk(node) {
    try {
      node.setPluginData("originalNodeId", node.id);
      stamped++;
    } catch (e) {
      console.warn("Failed stamping node:", node && node.id, e);
    }

    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) {
        walk(child);
      }
    }
  }

  walk(root);
  return stamped;
}

function collectClonedNodesByOriginalId(root) {
  const map = new Map();
  let mapped = 0;

  function walk(node) {
    try {
      const originalId = node.getPluginData("originalNodeId");
      if (originalId) {
        map.set(originalId, node);
        mapped++;
      }
    } catch (e) {
      console.warn("Failed collecting cloned node map entry:", node && node.id, e);
    }

    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) {
        walk(child);
      }
    }
  }

  walk(root);
  return { map, mapped };
}

function asArray(value) {
  return Array.isArray(value) ? value.filter(Boolean) : [];
}

function getNodeByPath(root, path) {
  if (!root || typeof path !== "string") return null;
  const trimmed = path.trim();
  if (!trimmed) return root;

  const segments = trimmed.split("/");
  let current = root;

  for (const segment of segments) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0) return null;
    if (!("children" in current) || !Array.isArray(current.children)) return null;
    if (index >= current.children.length) return null;
    current = current.children[index];
  }

  return current;
}

function getTopLevelNodeByPath(root, path) {
  if (!root || typeof path !== "string") return null;
  const trimmed = path.trim();
  if (!trimmed) return null;

  const first = trimmed.split("/")[0];
  const index = Number(first);
  if (!Number.isInteger(index) || index < 0) return null;
  if (!("children" in root) || !Array.isArray(root.children)) return null;
  if (index >= root.children.length) return null;
  return root.children[index];
}

function getSemanticName(item) {
  if (!item) return null;
  if (typeof item === "string") {
    const value = item.trim();
    return value || null;
  }
  if (typeof item === "object") {
    return item.semantic_name || item.semanticName || item.role || null;
  }
  return null;
}

function sanitizeLayerName(name) {
  return String(name)
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_а-яА-ЯёЁ:-]/g, "");
}

function setSemanticName(node, itemOrName) {
  const rawName = typeof itemOrName === "string" ? itemOrName : getSemanticName(itemOrName);
  if (!node || !rawName) return false;

  const clean = sanitizeLayerName(rawName);
  if (!clean) return false;

  node.name = clean;
  node.setPluginData("semanticName", clean);

  if (typeof itemOrName !== "string" && itemOrName && typeof itemOrName === "object") {
    if (itemOrName.role) node.setPluginData("semanticRole", String(itemOrName.role));
    if (itemOrName.source_figma_id) node.setPluginData("sourceFigmaId", String(itemOrName.source_figma_id));
    if (itemOrName.figma_node_id) node.setPluginData("sourceFigmaId", String(itemOrName.figma_node_id));
    if (itemOrName.confidence !== undefined) {
      node.setPluginData("semanticConfidence", String(itemOrName.confidence));
    }
  }

  return true;
}

function parseTargetSize(value, fallbackFrame) {
  const raw = String(value || "").trim();
  if (!raw || raw.toLowerCase() === "same") {
    return {
      width: Math.max(1, Math.round(fallbackFrame.width)),
      height: Math.max(1, Math.round(fallbackFrame.height)),
    };
  }
  const parts = raw.split(/[,\sx×]+/i).map((part) => Number(part.trim())).filter((n) => Number.isFinite(n) && n > 0);
  if (parts.length < 2) {
    throw new Error(`Target size must be widthxheight. Example: 1024x1280 (got ${raw})`);
  }
  return { width: Math.round(parts[0]), height: Math.round(parts[1]) };
}

function targetSizeName(targetResolution) {
  return `${Math.round(targetResolution.width)}x${Math.round(targetResolution.height)}`;
}

async function callBannerRawTargetPipeline(backendUrl, bannerPngBase64, rawJson, targetResolution) {
  const url = String(backendUrl || "").trim().replace(/\/+$/, "");
  if (!url) throw new Error("Backend URL is empty.");
  const response = await fetch(url + "/pipeline/banner-raw-to-target-json-json", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      banner_png_base64: bannerPngBase64,
      raw_json: rawJson,
      target_width: targetResolution.width,
      target_height: targetResolution.height,
      target_resolution: `${targetResolution.width}x${targetResolution.height}`,
      raw_frame_index: 0,
      top_k: 3,
      max_new_tokens: 64,
    }),
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = null;
  }
  if (!response.ok) {
    const detail =
      data && data.detail != null
        ? typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
        : text || `HTTP ${response.status}`;
    throw new Error(`Pipeline backend failed: ${detail}`);
  }
  if (!data || typeof data !== "object" || !data.final_json) {
    throw new Error("Pipeline backend returned invalid JSON or missing final_json.");
  }
  return data;
}

function resizeNodeIfPossible(node, width, height) {
  if (!node || !("resizeWithoutConstraints" in node)) return;
  try {
    node.resizeWithoutConstraints(Math.max(0.01, width), Math.max(0.01, height));
  } catch (e) {
    try {
      if ("resize" in node) node.resize(Math.max(0.01, width), Math.max(0.01, height));
    } catch (_e) {
      /* ignore unsupported nodes */
    }
  }
}

function jsonBounds(item) {
  const b = item && item.bounds && typeof item.bounds === "object" ? item.bounds : {};
  return {
    x: typeof b.x === "number" ? b.x : 0,
    y: typeof b.y === "number" ? b.y : 0,
    width: typeof b.width === "number" ? Math.max(1, b.width) : 1,
    height: typeof b.height === "number" ? Math.max(1, b.height) : 1,
  };
}

function figmaNodeTypeFromJson(item, isRoot) {
  const type = normalizeType(item && item.type ? item.type : "");
  const hasChildren = Array.isArray(item && item.children) && item.children.length > 0;
  if (isRoot || hasChildren || type === "frame" || type === "group") return "FRAME";
  if (type === "text") return "TEXT";
  return "RECTANGLE";
}

async function createNodeFromJsonItem(item, parent, parentBounds, isRoot) {
  if (!item || typeof item !== "object") return null;
  const bounds = jsonBounds(item);
  const figmaType = figmaNodeTypeFromJson(item, isRoot);
  let node;

  if (figmaType === "TEXT") {
    await figma.loadFontAsync({ family: "Inter", style: "Regular" });
    node = figma.createText();
    node.characters = String(item.characters || item.name || "Text");
    node.fontName = { family: "Inter", style: "Regular" };
    node.fontSize = typeof item.fontSize === "number" ? Math.max(1, item.fontSize) : 16;
    node.fills = [{ type: "SOLID", color: { r: 0.05, g: 0.06, b: 0.08 } }];
  } else if (figmaType === "FRAME") {
    node = figma.createFrame();
    node.layoutMode = "NONE";
    node.clipsContent = false;
    node.fills = isRoot ? [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }] : [];
  } else {
    node = figma.createRectangle();
    node.fills = [{ type: "SOLID", color: { r: 0.72, g: 0.78, b: 0.86 }, opacity: 0.28 }];
    node.strokes = [{ type: "SOLID", color: { r: 0.34, g: 0.45, b: 0.58 } }];
    node.strokeWeight = 1;
  }

  node.name = String(item.name || item.type || "json_node");
  if (!isRoot) {
    node.x = bounds.x - parentBounds.x;
    node.y = bounds.y - parentBounds.y;
  }
  resizeNodeIfPossible(node, bounds.width, bounds.height);
  parent.appendChild(node);

  const children = Array.isArray(item.children) ? item.children : [];
  for (const child of children) {
    await createNodeFromJsonItem(child, node, bounds, false);
  }
  return node;
}

async function drawJsonTreeBesideSelection(finalJson, sourceFrame, targetResolution) {
  if (!finalJson || typeof finalJson !== "object") {
    throw new Error("final_json must be an object.");
  }
  const rootBounds = jsonBounds(finalJson);
  const root = figma.createFrame();
  root.name = targetSizeName(targetResolution);
  root.x = sourceFrame.x + sourceFrame.width + 80;
  root.y = sourceFrame.y;
  root.layoutMode = "NONE";
  root.clipsContent = false;
  root.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  resizeNodeIfPossible(root, rootBounds.width, rootBounds.height);
  figma.currentPage.appendChild(root);

  const children = Array.isArray(finalJson.children) ? finalJson.children : [];
  for (const child of children) {
    await createNodeFromJsonItem(child, root, { x: 0, y: 0, width: rootBounds.width, height: rootBounds.height }, false);
  }
  return root;
}

function sameNumber(a, b, tolerance) {
  return Math.abs((Number(a) || 0) - (Number(b) || 0)) <= tolerance;
}

function frameMatchesSize(frame, bounds, tolerance) {
  if (!frame || !bounds) return false;
  return sameNumber(frame.width, bounds.width, tolerance) && sameNumber(frame.height, bounds.height, tolerance);
}

function normalizedName(value) {
  return String(value || "").trim().toLowerCase();
}

function findCandidateFrameInCurrentPage(selectedCandidate, finalJson, selectedFrame) {
  const candidate = selectedCandidate && typeof selectedCandidate === "object" ? selectedCandidate : {};
  const candidateId = String(candidate.id || (finalJson && finalJson.id) || "").trim();
  const candidateName = normalizedName(candidate.name || (finalJson && finalJson.name));
  const candidateBounds =
    candidate.bounds && typeof candidate.bounds === "object"
      ? candidate.bounds
      : finalJson && finalJson.bounds && typeof finalJson.bounds === "object"
        ? finalJson.bounds
        : null;

  if (candidateId) {
    try {
      const byId = figma.getNodeById(candidateId);
      if (byId && byId.type === "FRAME" && byId.id !== selectedFrame.id) {
        return { frame: byId, reason: "id" };
      }
    } catch (e) {
      console.warn("Candidate lookup by id failed:", candidateId, e);
    }
  }

  const frames = figma.currentPage.findAll((node) => node.type === "FRAME" && node.id !== selectedFrame.id);
  let best = null;
  let bestScore = -Infinity;
  for (const frame of frames) {
    const nameMatch = candidateName && normalizedName(frame.name) === candidateName;
    const sizeMatch = frameMatchesSize(frame, candidateBounds, 1);
    if (!nameMatch && !sizeMatch) continue;
    let score = 0;
    if (nameMatch) score += 100;
    if (sizeMatch) score += 50;
    if (candidateBounds) {
      score -= Math.abs(frame.width - (Number(candidateBounds.width) || 0)) / 1000;
      score -= Math.abs(frame.height - (Number(candidateBounds.height) || 0)) / 1000;
    }
    if (score > bestScore) {
      bestScore = score;
      best = frame;
    }
  }
  return best ? { frame: best, reason: "name/size" } : { frame: null, reason: "not_found" };
}

function scaleCloneTree(node, sx, sy, isRoot) {
  if (!node || typeof node !== "object") return 0;
  let scaled = 0;
  if (!isRoot) {
    if (typeof node.x === "number") node.x = node.x * sx;
    if (typeof node.y === "number") node.y = node.y * sy;
  }
  if ("width" in node && "height" in node) {
    resizeNodeIfPossible(node, node.width * sx, node.height * sy);
    scaled++;
  }
  if ("fontSize" in node && typeof node.fontSize === "number") {
    try {
      node.fontSize = Math.max(1, node.fontSize * Math.min(sx, sy));
    } catch (e) {
      console.warn("Failed scaling text font size:", node && node.name, e);
    }
  }
  if ("children" in node && Array.isArray(node.children)) {
    for (const child of node.children) {
      scaled += scaleCloneTree(child, sx, sy, false);
    }
  }
  return scaled;
}

function cloneCandidateFrameBesideSelection(candidateFrame, selectedFrame, finalJson, targetResolution) {
  const clone = candidateFrame.clone();
  clone.x = selectedFrame.x + selectedFrame.width + 80;
  clone.y = selectedFrame.y;
  clone.name = targetSizeName(targetResolution);
  const targetBounds = jsonBounds(finalJson || {});
  const sx = targetBounds.width / Math.max(1, candidateFrame.width);
  const sy = targetBounds.height / Math.max(1, candidateFrame.height);
  const scaled = scaleCloneTree(clone, sx, sy, true);
  resizeNodeIfPossible(clone, targetBounds.width, targetBounds.height);
  figma.currentPage.appendChild(clone);
  const summary = {
    applied: scaled,
    missing: [],
    scale_x: sx,
    scale_y: sy,
    source_width: candidateFrame.width,
    source_height: candidateFrame.height,
    target_width: targetBounds.width,
    target_height: targetBounds.height,
  };
  return { clone, summary };
}

function applyPredictedJsonToClone(predictedJson, convertedFrame) {
  const { map: nodeByOriginalId } = collectClonedNodesByOriginalId(convertedFrame);
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  let applied = 0;
  const missing = [];

  function resolve(item) {
    if (!item || typeof item !== "object") return null;
    const id = String(item.id || "").trim();
    const path = String(item.path || "").trim();
    if (id && nodeByOriginalId.has(id)) return nodeByOriginalId.get(id);
    if (path && pathNodeMap.has(path)) return pathNodeMap.get(path);
    return null;
  }

  function walk(item, isRoot) {
    if (!item || typeof item !== "object") return;
    const node = isRoot ? convertedFrame : resolve(item);
    const bounds = item.bounds && typeof item.bounds === "object" ? item.bounds : null;
    if (node) {
      if (!isRoot && bounds) {
        if (typeof bounds.x === "number") node.x = bounds.x;
        if (typeof bounds.y === "number") node.y = bounds.y;
        if (typeof bounds.width === "number" && typeof bounds.height === "number") {
          resizeNodeIfPossible(node, bounds.width, bounds.height);
        }
      } else if (isRoot && bounds && typeof bounds.width === "number" && typeof bounds.height === "number") {
        resizeNodeIfPossible(node, bounds.width, bounds.height);
      }
      if (item.name) node.name = String(item.name);
      applied++;
    } else if (item.id || item.path) {
      missing.push(String(item.id || item.path));
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) walk(child, false);
  }

  walk(predictedJson, true);
  return { applied, missing };
}

function buildPathNodeMap(root) {
  const map = new Map();
  map.set("", root);

  function walk(node, path) {
    if (!("children" in node) || !Array.isArray(node.children)) return;
    node.children.forEach((child, index) => {
      const childPath = path ? `${path}/${index}` : String(index);
      map.set(childPath, child);
      walk(child, childPath);
    });
  }

  walk(root, "");
  return map;
}

function getAncestorUnderRoot(root, node) {
  if (!root || !node) return null;
  let current = node;
  while (current && current.parent && current.parent.id !== root.id) {
    current = current.parent;
  }
  return current && current.parent && current.parent.id === root.id ? current : null;
}

function deriveContainerSemanticName(itemOrName) {
  const base = String(getSemanticName(itemOrName) || itemOrName || "").trim().toLowerCase();
  if (!base) return null;
  if (base === "headline") return "headline_group";
  if (base === "legal_text" || base === "legal") return "legal_group";
  if (
    base === "brand_name_yandex" ||
    base === "brand_name_lavka" ||
    base === "logo" ||
    base === "logo_heart" ||
    base === "logo_ellipse"
  ) {
    return "brand_group";
  }
  if (base === "age_badge") return "badge_group";
  if (base.indexOf("product_") === 0 || base.indexOf("hero_") === 0) return "hero_group";
  if (base === "decoration_star" || /^decoration_star(_\d+)?$/.test(base)) {
    return "decoration_star_group";
  }
  if (base.indexOf("decoration_") === 0) return "decoration_group";
  if (base.indexOf("background_") === 0) return "background_group";
  return null;
}

function isGenericLayerName(name) {
  const n = String(name || "").trim();
  if (!n) return true;
  if (/^\d+$/.test(n)) return true;
  if (/^(group|rectangle|vector|ellipse|line|polygon|star|frame|text)\s+\d+$/i.test(n)) return true;
  return false;
}

function collectGenericNodes(root) {
  const generic = [];
  function walk(node) {
    if (isGenericLayerName(node.name)) {
      generic.push({ id: node.id, name: node.name });
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) walk(child);
    }
  }
  walk(root);
  return generic;
}

function resolveNodeByBackendIdOrPath(item, nodeByOriginalId, pathNodeMap) {
  if (!item) return { node: null, matchedBy: "" };
  const id = String(item.source_figma_id || item.figma_node_id || item.node_id || item.id || "").trim();
  const path = String(item.path || "").trim();
  if (id && nodeByOriginalId.has(id)) return { node: nodeByOriginalId.get(id), matchedBy: "id" };
  if (path && pathNodeMap.has(path)) return { node: pathNodeMap.get(path), matchedBy: "path" };
  return { node: null, matchedBy: "" };
}

function canSafelyGroup(children) {
  if (!Array.isArray(children) || children.length < 2) return false;
  const parent = children[0].parent;
  if (!parent) return false;
  if (!children.every((child) => child.parent && child.parent.id === parent.id)) return false;
  if (parent.type === "INSTANCE" || parent.type === "COMPONENT" || parent.type === "COMPONENT_SET") return false;
  if (children.some((child) => "isMask" in child && child.isMask)) return false;
  return true;
}

function canSafelyUngroup(node) {
  if (!node || !("children" in node) || !Array.isArray(node.children)) return false;
  if (!isGenericLayerName(node.name)) return false;
  if (node.type === "INSTANCE" || node.type === "COMPONENT" || node.type === "COMPONENT_SET") return false;
  if (node.children.some((child) => "isMask" in child && child.isMask)) return false;
  return true;
}

function unwrapAnonymousWrappers(root) {
  let removed = 0;

  function walk(node) {
    if (!node || !("children" in node) || !Array.isArray(node.children)) return;
    for (const child of [...node.children]) {
      walk(child);
    }
    for (const child of [...node.children]) {
      if (!canSafelyUngroup(child)) continue;
      try {
        figma.ungroup(child);
        removed++;
      } catch (e) {
        console.warn("Failed ungrouping anonymous wrapper:", child && child.name, e);
      }
    }
  }

  walk(root);
  return removed;
}

function applySemanticFallbackName(node, groupName) {
  const base = String(groupName || "").toLowerCase();
  let fallback = "visual_asset";
  if (base.indexOf("logo") !== -1 || base.indexOf("brand") !== -1) fallback = "brand_text_part";
  else if (base.indexOf("decoration") !== -1) fallback = "decoration_part";
  else if (base.indexOf("background") !== -1) fallback = "background_part";
  else if (base.indexOf("hero") !== -1 || base.indexOf("product") !== -1) fallback = "logo_part";
  return setSemanticName(node, fallback);
}

function applyUpdatesWithVisibleContainers(convertedFrame, backendResponse, nodeByOriginalId, pathNodeMap) {
  const updates = asArray(backendResponse && backendResponse.updates);
  const sourceUpdateMap = new Map();
  let updatesAppliedById = 0;
  let updatesAppliedByPath = 0;
  let renamedVisibleContainers = 0;
  const missingSourceIds = new Set();
  const missingPaths = new Set();
  const explicitlyNamedNodeIds = new Set();

  function renameContainerForUpdate(update, exactNode) {
    const path = String(update.path || "").trim();
    const parentSemanticName = String(update.parent_semantic_name || "").trim();
    const containerName = parentSemanticName || deriveContainerSemanticName(update);
    if (!containerName) return;

    let containerNode = null;
    if (path) {
      const parentPath = path.indexOf("/") > -1 ? path.split("/").slice(0, -1).join("/") : "";
      containerNode = parentPath ? pathNodeMap.get(parentPath) : getTopLevelNodeByPath(convertedFrame, path);
    }
    if (!containerNode && exactNode && exactNode.parent) {
      containerNode = exactNode.parent.id === convertedFrame.id ? getAncestorUnderRoot(convertedFrame, exactNode) : exactNode.parent;
    }
    if (!containerNode || (exactNode && containerNode.id === exactNode.id)) return;

    if (setSemanticName(containerNode, containerName)) {
      renamedVisibleContainers++;
      explicitlyNamedNodeIds.add(containerNode.id);
    }
  }

  for (const update of updates) {
    const sourceId = String(update && update.source_figma_id ? update.source_figma_id : "").trim();
    const path = String(update && update.path ? update.path : "").trim();
    if (sourceId) sourceUpdateMap.set(sourceId, update);

    if (!getSemanticName(update)) {
      if (sourceId) missingSourceIds.add(sourceId);
      if (path) missingPaths.add(path);
      continue;
    }

    const resolved = resolveNodeByBackendIdOrPath(update, nodeByOriginalId, pathNodeMap);
    if (resolved.node && resolved.node.id !== convertedFrame.id) {
      if (setSemanticName(resolved.node, update)) {
        explicitlyNamedNodeIds.add(resolved.node.id);
        if (resolved.matchedBy === "id") updatesAppliedById++;
        else updatesAppliedByPath++;
      }
      renameContainerForUpdate(update, resolved.node);
    } else {
      if (sourceId) missingSourceIds.add(sourceId);
      if (path) missingPaths.add(path);
    }
  }

  return {
    updates,
    sourceUpdateMap,
    updatesAppliedById,
    updatesAppliedByPath,
    renamedVisibleContainers,
    missingSourceIds,
    missingPaths,
    explicitlyNamedNodeIds
  };
}

function applySemanticsToClone(result, convertedFrame) {
  const preservedRootName = convertedFrame.name;
  const { map: nodeByOriginalId, mapped } = collectClonedNodesByOriginalId(convertedFrame);
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  const updatesSummary = applyUpdatesWithVisibleContainers(convertedFrame, result, nodeByOriginalId, pathNodeMap);
  const semanticElements = asArray(result && result.semantic && result.semantic.elements);
  const semanticGroups = asArray(result && result.semantic && result.semantic.groups);
  const elementById = new Map();

  semanticElements.forEach((el) => {
    const id = String(
      el && (el.figma_node_id || el.source_figma_id || el.node_id || el.id)
        ? (el.figma_node_id || el.source_figma_id || el.node_id || el.id)
        : ""
    ).trim();
    if (id) elementById.set(id, el);
  });

  let semanticElementsApplied = 0;
  let semanticGroupsApplied = 0;
  let groupsCreated = 0;

  for (const element of semanticElements) {
    const id = String(
      element && (element.figma_node_id || element.source_figma_id || element.node_id || element.id)
        ? (element.figma_node_id || element.source_figma_id || element.node_id || element.id)
        : ""
    ).trim();

    const resolved = resolveNodeByBackendIdOrPath(element, nodeByOriginalId, pathNodeMap);
    if (resolved.node && resolved.node.id !== convertedFrame.id && setSemanticName(resolved.node, element)) {
      semanticElementsApplied++;
      updatesSummary.explicitlyNamedNodeIds.add(resolved.node.id);
    } else {
      if (id) updatesSummary.missingSourceIds.add(id);
      if (element && element.path) updatesSummary.missingPaths.add(String(element.path));
    }
  }

  for (const group of semanticGroups) {
    const groupName = getSemanticName(group);
    const children = asArray(group && group.children);
    if (!groupName || children.length === 0) continue;

    let groupApplied = false;
    const knownContainerId = String(
      group && (group.figma_node_id || group.source_figma_id || group.node_id || group.source_node_id)
        ? (group.figma_node_id || group.source_figma_id || group.node_id || group.source_node_id)
        : ""
    ).trim();
    if (knownContainerId && nodeByOriginalId.has(knownContainerId)) {
      const containerNode = nodeByOriginalId.get(knownContainerId);
      if (containerNode.id !== convertedFrame.id && setSemanticName(containerNode, group)) {
        updatesSummary.renamedVisibleContainers++;
        updatesSummary.explicitlyNamedNodeIds.add(containerNode.id);
        groupApplied = true;
      }
    }

    const matchedChildren = [];
    const parentMap = new Map();
    for (const childRaw of children) {
      const childItem = typeof childRaw === "object" ? childRaw : { source_figma_id: childRaw };
      const childId = String(
        childItem && (childItem.source_figma_id || childItem.figma_node_id || childItem.node_id || childItem.id)
          ? (childItem.source_figma_id || childItem.figma_node_id || childItem.node_id || childItem.id)
          : ""
      ).trim();

      const resolvedChild = resolveNodeByBackendIdOrPath(childItem, nodeByOriginalId, pathNodeMap);
      const childNode = resolvedChild.node;
      if (!childNode) {
        if (childId) updatesSummary.missingSourceIds.add(childId);
        continue;
      }

      matchedChildren.push(childNode);
      if (childNode.parent) {
        parentMap.set(childNode.parent.id, childNode.parent);
      }

      const childElement = childId ? elementById.get(childId) : null;
      const childUpdate = childId ? updatesSummary.sourceUpdateMap.get(childId) : null;

      if (childElement && getSemanticName(childElement)) {
        setSemanticName(childNode, childElement);
        updatesSummary.explicitlyNamedNodeIds.add(childNode.id);
      } else if (childUpdate && getSemanticName(childUpdate)) {
        setSemanticName(childNode, childUpdate);
        updatesSummary.explicitlyNamedNodeIds.add(childNode.id);
      } else if (isGenericLayerName(childNode.name)) {
        applySemanticFallbackName(childNode, groupName);
      }

      const childPath = String(childItem && childItem.path ? childItem.path : "").trim();
      if (childPath) {
        const topLevelNode = getTopLevelNodeByPath(convertedFrame, childPath);
        if (topLevelNode && setSemanticName(topLevelNode, group)) {
          updatesSummary.renamedVisibleContainers++;
          updatesSummary.explicitlyNamedNodeIds.add(topLevelNode.id);
          groupApplied = true;
        }
      }
    }

    if (canSafelyGroup(matchedChildren)) {
      try {
        const groupNode = figma.group(matchedChildren, matchedChildren[0].parent);
        if (setSemanticName(groupNode, group)) {
          groupsCreated++;
          groupApplied = true;
        }
      } catch (e) {
        console.warn("Failed creating semantic group:", e);
      }
    } else if (parentMap.size === 1) {
      const existingContainer = Array.from(parentMap.values())[0];
      if (existingContainer && existingContainer.id !== convertedFrame.id && setSemanticName(existingContainer, group)) {
        updatesSummary.renamedVisibleContainers++;
        updatesSummary.explicitlyNamedNodeIds.add(existingContainer.id);
        groupApplied = true;
      }
    }

    if (groupApplied) semanticGroupsApplied++;
  }

  const remainingGeneric = collectGenericNodes(convertedFrame);
  convertedFrame.name = preservedRootName;

  return {
    mapped,
    updatesReceived: updatesSummary.updates.length,
    updatesAppliedById: updatesSummary.updatesAppliedById,
    updatesAppliedByPath: updatesSummary.updatesAppliedByPath,
    semanticElementsReceived: semanticElements.length,
    semanticElementsApplied,
    semanticGroupsReceived: semanticGroups.length,
    semanticGroupsApplied,
    renamedVisibleContainers: updatesSummary.renamedVisibleContainers,
    groupsCreated,
    groupsSkipped: Math.max(0, semanticGroups.length - semanticGroupsApplied),
    missingSourceIds: Array.from(updatesSummary.missingSourceIds),
    missingPaths: Array.from(updatesSummary.missingPaths),
    remainingGeneric
  };
}

function applyFinalJsonToClone(finalJson, convertedFrame) {
  const preservedRootName = convertedFrame.name;
  const { map: nodeByOriginalId, mapped } = collectClonedNodesByOriginalId(convertedFrame);
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  let renamed = 0;
  let groupsCreated = 0;
  let groupsRenamed = 0;
  let wrappersRemoved = 0;
  const missing = [];

  function resolveFinalNode(item) {
    if (!item || typeof item !== "object") return null;
    const id = String(item.id || "").trim();
    const path = String(item.path || "").trim();
    if (id && nodeByOriginalId.has(id)) return nodeByOriginalId.get(id);
    if (path && pathNodeMap.has(path)) return pathNodeMap.get(path);
    return null;
  }

  function semanticLabel(item) {
    return String(item && (item.name || item.role || item.type) ? (item.name || item.role || item.type) : "");
  }

  function isGroupSemanticName(value) {
    return /_group$/.test(String(value || ""));
  }

  function isTopLevelSemanticGroupName(value) {
    return new Set([
      "brand_group",
      "headline_group",
      "legal_text_group",
      "age_badge_group",
      "hero_image_group",
      "star_group",
      "glow_group",
      "bg_shape_group",
    ]).has(String(value || ""));
  }

  function promoteAncestorGroup(node, name) {
    if (!node || !name) return null;
    let current = node;
    while (current && current.parent && current.parent.id !== convertedFrame.id) {
      current = current.parent;
    }
    const top = current && current.parent && current.parent.id === convertedFrame.id ? current : null;
    if (!top || top.id === convertedFrame.id) return null;
    setSemanticName(top, name);
    return top;
  }

  function applyTree(item) {
    if (!item || typeof item !== "object") return null;
    const children = Array.isArray(item.children) ? item.children : [];
    const exactNode = resolveFinalNode(item);
    const label = semanticLabel(item);

    if (exactNode && exactNode.id !== convertedFrame.id) {
      if (setSemanticName(exactNode, label)) renamed++;
      for (const child of children) applyTree(child);
      return exactNode;
    }

    const path = String(item.path || "").trim();
    if (!exactNode && path) {
      const wrapper = path.indexOf("/") > -1
        ? pathNodeMap.get(path.split("/").slice(0, -1).join("/"))
        : getTopLevelNodeByPath(convertedFrame, path);
      if (wrapper && wrapper.id !== convertedFrame.id && (isGenericLayerName(wrapper.name) || isTopLevelSemanticGroupName(label))) {
        setSemanticName(wrapper, label || "semantic_group");
        groupsRenamed++;
      }
    }

    const matchedChildren = [];
    for (const child of children) {
      const childNode = applyTree(child);
      if (childNode) matchedChildren.push(childNode);
    }

    if (matchedChildren.length === 0) {
      if (item.name && item.id) missing.push(String(item.id));
      return null;
    }

    if (isTopLevelSemanticGroupName(label)) {
      const promoted = promoteAncestorGroup(matchedChildren[0], label);
      if (promoted) {
        groupsRenamed++;
        return promoted;
      }
    }

    const topLevelMap = new Map();
    for (const node of matchedChildren) {
      const top = getAncestorUnderRoot(convertedFrame, node);
      if (top && top.id !== convertedFrame.id) {
        topLevelMap.set(top.id, top);
      }
    }
    if (topLevelMap.size === 1) {
      const topWrapper = Array.from(topLevelMap.values())[0];
      if (topWrapper && (isGenericLayerName(topWrapper.name) || isTopLevelSemanticGroupName(label))) {
        setSemanticName(topWrapper, label || "semantic_group");
        groupsRenamed++;
        wrappersRemoved += unwrapAnonymousWrappers(topWrapper);
        return topWrapper;
      }
    }

    if (canSafelyGroup(matchedChildren)) {
      try {
        const groupNode = figma.group(matchedChildren, matchedChildren[0].parent);
        if (setSemanticName(groupNode, label || "semantic_group")) {
          groupsCreated++;
          wrappersRemoved += unwrapAnonymousWrappers(groupNode);
          return groupNode;
        }
      } catch (e) {
        console.warn("Failed grouping final_json children:", item && item.name, e);
      }
    }

    const parentMap = new Map();
    for (const node of matchedChildren) {
      if (node.parent) parentMap.set(node.parent.id, node.parent);
    }
    if (parentMap.size === 1) {
      const parent = Array.from(parentMap.values())[0];
      if (parent && parent.id !== convertedFrame.id && setSemanticName(parent, label || "semantic_group")) {
        groupsRenamed++;
        wrappersRemoved += unwrapAnonymousWrappers(parent);
        return parent;
      }
    }

    return matchedChildren[0];
  }

  for (const child of Array.isArray(finalJson && finalJson.children) ? finalJson.children : []) {
    applyTree(child);
  }
  wrappersRemoved += unwrapAnonymousWrappers(convertedFrame);

  convertedFrame.name = preservedRootName;
  return {
    mapped,
    renamed,
    groupsCreated,
    groupsRenamed,
    wrappersRemoved,
    missing,
    remainingGeneric: collectGenericNodes(convertedFrame),
  };
}

function getSelectionInfo() {
  const selection = figma.currentPage.selection;

  if (selection.length === 0) {
    return { hasSelection: false };
  }

  const node = selection[0];

  return {
    hasSelection: true,
    id: node.id,
    name: node.name,
    type: node.type,
    isFrame: node.type === "FRAME",
    width: "width" in node ? Number(node.width.toFixed(2)) : null,
    height: "height" in node ? Number(node.height.toFixed(2)) : null
  };
}

function sendSelectionInfo() {
  figma.ui.postMessage({
    type: "selection-info",
    selection: getSelectionInfo()
  });
}

figma.on("selectionchange", () => {
  sendSelectionInfo();
});

function postStatus(message) {
  figma.ui.postMessage({ type: "status", message });
}

function postError(message) {
  figma.ui.postMessage({ type: "error", message });
}

figma.ui.onmessage = async (msg) => {
  if (msg.type === "export-selected-frame-html-css") {
    const selection = figma.currentPage.selection;
    if (selection.length !== 1 || selection[0].type !== "FRAME") {
      postError("Select exactly one frame.");
      figma.ui.postMessage({ type: "html-css-export-result", ok: false });
      sendSelectionInfo();
      return;
    }

    try {
      const selectedFrame = selection[0];
      postStatus(`HTML/CSS export: serializing ${selectedFrame.name}...`);
      const origin = getOrigin(selectedFrame);
      const rawJson = serializeNode(selectedFrame, origin, "");
      rawJson.templateId = "figma_plugin_html_css_export";
      postStatus(`HTML/CSS export: exporting full banner render...`);
      const bannerPngBytes = await exportFramePngBytes(selectedFrame);
      const bannerPngBase64 = uint8ToBase64(bannerPngBytes);
      postStatus(`HTML/CSS export: exporting inspectable element assets...`);
      const elementAssets = await exportElementAssetsForHtml(
        selectedFrame,
        rawJson,
        MAX_ELEMENT_LAYER_PNGS,
      );
      const html = rawJsonToHtmlCss(rawJson, bannerPngBase64, elementAssets);
      figma.ui.postMessage({
        type: "html-css-export-result",
        ok: true,
        html,
        fileName: `${selectedFrame.name || "figma-export"}-${Math.round(selectedFrame.width)}x${Math.round(selectedFrame.height)}`,
      });
      postStatus(`HTML/CSS export ready (${elementAssets.length} element assets, ${html.length.toLocaleString()} chars).`);
    } catch (err) {
      console.error("HTML/CSS export failed:", err);
      postError(String(err && err.message ? err.message : err));
      figma.ui.postMessage({ type: "html-css-export-result", ok: false });
    }
    return;
  }

  if (msg.type === "pipeline-target-json-selected-frame") {
    const selection = figma.currentPage.selection;
    if (selection.length !== 1 || selection[0].type !== "FRAME") {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Select exactly one frame.");
      sendSelectionInfo();
      return;
    }

    const selectedFrame = selection[0];
    const backendUrl = String(msg.backendUrl || "").trim();
    if (!backendUrl) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Backend URL is empty.");
      return;
    }

    try {
      figma.ui.postMessage({ type: "pipeline-busy", busy: true });
      const targetResolution = parseTargetSize(msg.targetSize, selectedFrame);
      postStatus(`Pipeline: stamping original node ids for ${selectedFrame.name}...`);
      const stampedNodeCount = stampOriginalNodeIds(selectedFrame);

      postStatus("Pipeline: serializing selected Figma design...");
      const origin = getOrigin(selectedFrame);
      const rawJson = serializeNode(selectedFrame, origin, "");
      rawJson.templateId = "figma_plugin_pipeline_source";

      postStatus("Pipeline: exporting banner PNG...");
      const pngBytes = await exportFramePngBytes(selectedFrame);
      const pngBase64 = uint8ToBase64(pngBytes);

      postStatus(
        `Pipeline: calling backend for ${targetResolution.width}×${targetResolution.height} target JSON...`,
      );
      const result = await callBannerRawTargetPipeline(
        backendUrl,
        pngBase64,
        rawJson,
        targetResolution,
      );

      const selectedGuide = result.selected_candidate || {};
      postStatus(`Pipeline: finding candidate frame "${selectedGuide.name || "unknown"}" in current page...`);
      const lookup = findCandidateFrameInCurrentPage(selectedGuide, result.final_json, selectedFrame);
      let convertedFrame;
      let drawMode;
      let applySummary = null;
      if (lookup.frame) {
        postStatus(`Pipeline: cloning matched candidate frame (${lookup.reason}) and scaling to target size...`);
        const cloned = cloneCandidateFrameBesideSelection(
          lookup.frame,
          selectedFrame,
          result.final_json,
          targetResolution,
        );
        convertedFrame = cloned.clone;
        applySummary = cloned.summary;
        drawMode = "clone_matched_candidate_frame_scaled";
      } else {
        postStatus("Pipeline: candidate frame not found in current page; drawing returned JSON fallback...");
        convertedFrame = await drawJsonTreeBesideSelection(result.final_json, selectedFrame, targetResolution);
        drawMode = "create_from_returned_json_fallback";
      }

      figma.currentPage.selection = [convertedFrame];
      figma.viewport.scrollAndZoomIntoView([selectedFrame, convertedFrame]);

      figma.notify(
        `Target clone created.\n` +
          `Qwen class: ${result.category}\n` +
          `Guide: ${selectedGuide.name || "unknown"}\n` +
          `Mode: ${drawMode}\n` +
          `Selected stamped nodes: ${stampedNodeCount}\n` +
          (applySummary
            ? `Scaled candidate nodes: ${applySummary.applied}\nScale: ${applySummary.scale_x.toFixed(3)} × ${applySummary.scale_y.toFixed(3)}`
            : `Fallback drawn from JSON`),
        { timeout: 8 },
      );

      figma.ui.postMessage({ type: "done" });
      sendSelectionInfo();
    } catch (err) {
      console.error("Pipeline target JSON failed:", err);
      var pipelineMsg =
        err && err.stack
          ? err.message + "\n\n" + err.stack
          : String(err && err.message ? err.message : err);
      if (err && err.message === "Failed to fetch") {
        pipelineMsg +=
          "\n\nFigma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. " +
          "Make sure Backend URL exactly matches the manifest, then reload the development plugin.";
      }
      postError(pipelineMsg);
      sendSelectionInfo();
    } finally {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
    }
    return;
  }

  if (msg.type === "semantic-json-grid-selected-frame") {
    const selection = figma.currentPage.selection;
    if (selection.length !== 1 || selection[0].type !== "FRAME") {
      postError("Select exactly one frame.");
      sendSelectionInfo();
      return;
    }

    const selectedFrame = selection[0];
    const backendUrl = String(msg.backendUrl || "").trim().replace(/\/+$/, "");
    if (!backendUrl) {
      postError("Backend URL is empty.");
      return;
    }

    const maxNewTokens = Math.min(4096, Math.max(256, parseInt(String(msg.maxNewTokens || "4096"), 10) || 4096));

    try {
      postStatus("Semantic JSON: stamping node ids…");
      stampOriginalNodeIds(selectedFrame);

      postStatus("Semantic JSON: serializing raw JSON…");
      const origin = getOrigin(selectedFrame);
      const rawJson = serializeNode(selectedFrame, origin, "");
      rawJson.templateId = "figma_plugin_semantic_grid";

      postStatus("Semantic JSON: exporting banner PNG…");
      const bannerBytes = await exportFramePngBytes(selectedFrame);

      postStatus("Semantic JSON: building element grid PNG…");
      const { atlasPngBytes, regions, atlasSize } = await buildElementAtlasPngAndRegions(
        selectedFrame,
        MAX_ELEMENT_LAYER_PNGS,
      );
      if (!atlasPngBytes || !atlasPngBytes.length) {
        throw new Error("Grid atlas export failed (empty PNG).");
      }
      if (!regions.length) {
        throw new Error("Grid atlas has no regions (no packable leaf elements).");
      }
      injectAtlasRegionsIntoRawJson(rawJson, regions);
      attachAtlasMetadataToRawJson(rawJson, atlasSize, regions);

      const requestUrl = `${backendUrl}/figma/convert-semantic-json`;
      postStatus("Semantic JSON: calling backend + Qwen (may take several minutes)…");

      const jsonStr = JSON.stringify(rawJson);
      const bannerU8 =
        bannerBytes instanceof Uint8Array ? bannerBytes : new Uint8Array(bannerBytes);

      const boundary =
        "----figmaSemantic" +
        Math.random().toString(36).slice(2) +
        Date.now().toString(36) +
        Math.random().toString(36).slice(2);
      const mpBody = buildMultipartFormDataBody(boundary, [
        { name: "banner", filename: "banner.png", contentType: "image/png", body: bannerU8 },
        { name: "grid", filename: "elements.png", contentType: "image/png", body: atlasPngBytes },
        {
          name: "raw_json",
          filename: "raw.json",
          contentType: "application/json; charset=utf-8",
          body: utf8Bytes(jsonStr),
        },
        {
          name: "max_new_tokens",
          filename: null,
          contentType: "text/plain; charset=utf-8",
          body: utf8Bytes(String(maxNewTokens)),
        },
      ]);

      const response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
        body: mpBody,
      });
      const responseText = await response.text();
      let data = null;
      try {
        data = responseText ? JSON.parse(responseText) : null;
      } catch (_parseErr) {
        data = null;
      }

      if (!response.ok) {
        const detail =
          data && typeof data === "object" && data.detail != null
            ? typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail)
            : responseText || "HTTP " + String(response.status);
        throw new Error(detail);
      }

      if (!data || typeof data !== "object" || !("semantic_json" in data)) {
        throw new Error("Backend response missing semantic_json.");
      }

      const pretty = JSON.stringify(data.semantic_json, null, 2);
      figma.ui.postMessage({
        type: "semantic-json-result",
        ok: true,
        jsonText: pretty,
        fileName: `${selectedFrame.name || "semantic"}-${Math.round(selectedFrame.width)}x${Math.round(selectedFrame.height)}`,
      });
      postStatus("Semantic JSON: done.");
      figma.ui.postMessage({ type: "done" });
      sendSelectionInfo();
    } catch (err) {
      console.error("Semantic JSON grid flow failed:", err);
      let errText =
        err && err.stack ? err.message + "\n\n" + err.stack : String(err && err.message ? err.message : err);
      if (err && err.message === "Failed to fetch") {
        errText +=
          "\n\nFigma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. " +
          "Match Backend URL exactly, then reload the plugin.";
      }
      postError(errText);
      figma.ui.postMessage({
        type: "semantic-json-result",
        ok: false,
        error: err && err.message ? err.message : String(err),
      });
      sendSelectionInfo();
    }
    return;
  }
};

sendSelectionInfo();