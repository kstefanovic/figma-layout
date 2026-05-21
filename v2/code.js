/**
 * Figma plugin main thread. Flows:
 * - ``POST …/pipeline/banner-raw-to-target-json-json`` — banner + raw JSON + target size → classify into class 1..6, retrieve best template in that class, return target JSON; plugin draws clone beside the source.
 * - ``POST …/api/layout-transformer-v2`` — selected rich semantic JSON + target size → V2 Layout Transformer output; plugin clones beside the original and applies returned ``final_json``.
 * - ``POST …/figma/convert-semantic-json`` — banner + grid PNG + raw JSON → Qwen (multipart sent from the plugin UI iframe with ``FormData``, same as ``frontend/figma.html``); server merges ``{names:{id:…}}`` into full semantic JSON; main thread clones beside the original, reparents to match JSON hierarchy, then renames from that JSON.
 * - HTML/CSS export from serialized JSON + assets (local).
 * - Rich JSON export — full ``serializeNode`` tree for selected frame(s), downloaded from the UI.
 */
figma.showUI(__html__, { width: 400, height: 760 });

/** Horizontal gap between the source frame and a sibling created by the plugin (px). */
const BESIDE_FRAME_GAP = 80;
const SOURCE_STYLE_TEXT_ROLES = new Set([
  "headline",
  "subheadline_delivery_time",
  "legal_text",
  "age_badge",
]);

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

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function jsonSafeValue(value) {
  if (value === figma.mixed || typeof value === "symbol" || typeof value === "function") return undefined;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (_e) {
    return undefined;
  }
}

function copySerializableNodeProperty(target, node, key) {
  if (!(key in node)) return;
  const value = jsonSafeValue(node[key]);
  if (value !== undefined) target[key] = value;
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
    if ("fontSize" in node && finiteNumber(node.fontSize)) base.fontSize = node.fontSize;
    copySerializableNodeProperty(base, node, "fontName");
    if ("textAlignHorizontal" in node) base.textAlignHorizontal = node.textAlignHorizontal;
    if ("textAlignVertical" in node) base.textAlignVertical = node.textAlignVertical;
    if ("textAutoResize" in node) base.textAutoResize = node.textAutoResize;
    for (const key of [
      "fills",
      "lineHeight",
      "letterSpacing",
      "paragraphSpacing",
      "paragraphIndent",
      "textCase",
      "textDecoration",
      "textStyleId",
      "hyperlink",
    ]) {
      copySerializableNodeProperty(base, node, key);
    }
  }

  for (const key of ["fills", "strokes", "strokeWeight", "effects", "blendMode", "cornerRadius"]) {
    copySerializableNodeProperty(base, node, key);
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

/** Banner for semantic JSON: cap longest side so Figma export + upload stay smaller (backend may downscale again). */
const SEMANTIC_BANNER_EXPORT_MAX_EDGE = 1024;

async function exportFramePngBytesMaxLongEdge(node, maxLongEdge) {
  const cap = Math.max(64, Number(maxLongEdge) || 1024);
  const w = "width" in node ? Number(node.width) : 1;
  const h = "height" in node ? Number(node.height) : 1;
  const longest = Math.max(1, w, h);
  const scale = longest <= cap ? 1 : cap / longest;
  return await node.exportAsync({
    format: "PNG",
    constraint: { type: "SCALE", value: scale },
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

/** Max leaves packed into the element atlas (lower = faster layout/export; fewer cells for Qwen). */
const MAX_ELEMENT_LAYER_PNGS = 256;

/** Gap between atlas cells (px). Smaller = much faster Figma layout/raster for large trees. */
const ATLAS_GAP = 20;
const ATLAS_CELL_PADDING = 8;
const ATLAS_MAX_ROW_WIDTH = 8192;
/** Max clone width/height per cell before pack (smaller = faster clone/render). */
const ATLAS_MAX_CELL = 2048;
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
 * Clone leaves into one off-screen frame, pack in rows with modest spacing, draw visible
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
  let duplicates = 0;

  function walk(node) {
    try {
      const originalId = node.getPluginData("originalNodeId");
      if (originalId) {
        if (map.has(originalId)) {
          duplicates++;
        } else {
          map.set(originalId, node);
        }
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
  if (duplicates > 0) {
    console.warn("[original-id-duplicates]", duplicates);
  }
  return { map, mapped, duplicates };
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
    const direct = item.name != null ? String(item.name).trim() : "";
    if (direct) return direct;
    return item.semantic_name || item.semanticName || item.role || null;
  }
  return null;
}

function semanticRoleName(item) {
  return String(item && (item.name || item.semantic_name || item.semanticName || item.role) || "").trim();
}

function shouldCloneSourceTextForRole(item) {
  return false;
}

function hasImageFill(node) {
  const fills = node && Array.isArray(node.fills) ? node.fills : [];
  return fills.some((fill) => fill && fill.type === "IMAGE");
}

function hasGradientFill(node) {
  const fills = node && Array.isArray(node.fills) ? node.fills : [];
  return fills.some((fill) => fill && String(fill.type || "").indexOf("GRADIENT_") === 0);
}

function sourceFrameHasGradient(sourceFrame) {
  let found = false;

  function walk(node) {
    if (!node || found) return;
    if (hasGradientFill(node) || /^background_gradient_/.test(String(node.name || "").trim())) {
      found = true;
      return;
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) walk(child);
    }
  }

  walk(sourceFrame);
  return found;
}

function collectSourceContentMap(sourceFrame) {
  const textByRole = new Map();
  const textById = new Map();
  const textByPath = new Map();
  const imageFillsByRole = new Map();
  const imageFillsById = new Map();
  const imageFillsByPath = new Map();
  const visualNodeByRole = new Map();
  const visualNodeById = new Map();
  const visualNodeByPath = new Map();

  function setIfPresent(map, key, value) {
    const normalizedKey = String(key || "").trim();
    if (normalizedKey) map.set(normalizedKey, value);
  }

  function pluginData(node, key) {
    try {
      return typeof node.getPluginData === "function" ? node.getPluginData(key) : "";
    } catch (_e) {
      return "";
    }
  }

  function walk(node, path) {
    if (!node) return;
    const role =
      String(node.name || "").trim() ||
      pluginData(node, "semanticName") ||
      pluginData(node, "semanticRole");
    if ("characters" in node) {
      const characters = node.characters;
      setIfPresent(textByRole, role, characters);
      setIfPresent(textById, node.id, characters);
      setIfPresent(textById, pluginData(node, "originalNodeId"), characters);
      setIfPresent(textByPath, path, characters);
    }
    if (hasImageFill(node)) {
      const imageFills = jsonSafeValue(node.fills);
      setIfPresent(imageFillsByRole, role, imageFills);
      setIfPresent(imageFillsById, node.id, imageFills);
      setIfPresent(imageFillsById, pluginData(node, "originalNodeId"), imageFills);
      setIfPresent(imageFillsByPath, path, imageFills);
    }
    const hasChildren = "children" in node && Array.isArray(node.children) && node.children.length > 0;
    const isVisualContentNode = !("characters" in node) && ("fills" in node || hasChildren);
    const type = String(node.type || "").toUpperCase();
    const cloneableVisual =
      isVisualContentNode &&
      node !== sourceFrame &&
      (
        type === "VECTOR" ||
        type === "STAR" ||
        type === "BOOLEAN_OPERATION" ||
        type === "RECTANGLE" ||
        hasImageFill(node) ||
        role === "logo"
      );
    if (cloneableVisual) {
      setIfPresent(visualNodeByRole, role, node);
      setIfPresent(visualNodeById, node.id, node);
      setIfPresent(visualNodeById, pluginData(node, "originalNodeId"), node);
      setIfPresent(visualNodeByPath, path, node);
    }
    if ("children" in node && Array.isArray(node.children)) {
      node.children.forEach((child, index) => {
        walk(child, path ? `${path}/${index}` : String(index));
      });
    }
  }

  walk(sourceFrame, "");
  return {
    textByRole,
    textById,
    textByPath,
    imageFillsByRole,
    imageFillsById,
    imageFillsByPath,
    visualNodeByRole,
    visualNodeById,
    visualNodeByPath,
  };
}

function collectSourceTextContentMap(sourceFrame) {
  const sourceContentMap = collectSourceContentMap(sourceFrame);
  return {
    byRole: sourceContentMap.textByRole,
    byId: sourceContentMap.textById,
    byPath: sourceContentMap.textByPath,
  };
}

const LOGO_PART_ROLES = new Set(["logo_back", "logo_fore"]);

/**
 * Index ids and semantic roles present on the selected source frame (authoritative allow-list).
 */
function collectSourceElementIndex(sourceFrame) {
  const ids = new Set();
  const roles = new Set();
  let hasGradient = false;

  function pluginData(node, key) {
    try {
      return typeof node.getPluginData === "function" ? node.getPluginData(key) : "";
    } catch (_e) {
      return "";
    }
  }

  function walk(node) {
    if (!node) return;
    if (hasGradientFill(node)) hasGradient = true;
    const id = String(node.id || "").trim();
    if (id) ids.add(id);
    const storedId = String(pluginData(node, "originalNodeId") || "").trim();
    if (storedId) ids.add(storedId);
    const role =
      String(node.name || "").trim() ||
      String(pluginData(node, "semanticName") || "").trim() ||
      String(pluginData(node, "semanticRole") || "").trim();
    if (role) roles.add(role);
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) walk(child);
    }
  }

  walk(sourceFrame);
  return { ids, roles, hasGradient };
}

function jsonItemAllowedInSource(item, sourceIndex) {
  if (!item || !sourceIndex) return false;
  const id = String(item.id || "").trim();
  if (id && sourceIndex.ids.has(id)) return true;
  const role = semanticRoleName(item);
  if (!role) return false;
  if (sourceIndex.roles.has(role)) return true;
  if (sourceIndex.hasGradient && /^background_gradient_/.test(role)) return true;
  if (LOGO_PART_ROLES.has(role) && sourceIndex.roles.has("logo")) return true;
  return false;
}

function isSyntheticGroupRole(role) {
  return /_group$/.test(String(role || ""));
}

function reindexFinalJsonPaths(item, path) {
  if (!item || typeof item !== "object") return;
  item.path = path;
  const children = Array.isArray(item.children) ? item.children : [];
  for (let i = 0; i < children.length; i++) {
    const childPath = path ? `${path}/${i}` : String(i);
    reindexFinalJsonPaths(children[i], childPath);
  }
}

/**
 * Drop backend-only layers so the generated banner never contains elements absent from the source.
 */
function filterFinalJsonToSourceElements(finalJson, sourceFrame) {
  const sourceIndex = collectSourceElementIndex(sourceFrame);
  const removedRoles = [];

  function filterChildren(children) {
    const kept = [];
    for (const child of Array.isArray(children) ? children : []) {
      const filtered = filterNode(child);
      if (!filtered) {
        removedRoles.push(semanticRoleName(child) || String(child.id || "unknown"));
        continue;
      }
      const role = semanticRoleName(child);
      if (
        isSyntheticGroupRole(role) &&
        !sourceIndex.roles.has(role) &&
        Array.isArray(filtered.children) &&
        filtered.children.length
      ) {
        removedRoles.push(role);
        for (const grandchild of filtered.children) kept.push(grandchild);
      } else {
        kept.push(filtered);
      }
    }
    return kept;
  }

  function filterNode(item) {
    if (!item || typeof item !== "object") return null;
    if (!jsonItemAllowedInSource(item, sourceIndex)) return null;
    const copy = {};
    for (const key of Object.keys(item)) {
      if (key !== "children") copy[key] = item[key];
    }
    copy.children = filterChildren(item.children);
    const role = semanticRoleName(item);
    if (isSyntheticGroupRole(role) && !sourceIndex.roles.has(role) && copy.children.length === 0) {
      return null;
    }
    return copy;
  }

  const filteredRoot = filterNode(finalJson);
  const json =
    filteredRoot ||
    Object.assign({}, finalJson, {
      children: filterChildren(finalJson.children),
    });
  reindexFinalJsonPaths(json, "");
  const report = {
    json,
    removedRoles,
    removedCount: removedRoles.length,
    sourceRoleCount: sourceIndex.roles.size,
    sourceIdCount: sourceIndex.ids.size,
  };
  console.log("[SOURCE_ELEMENT_FILTER]", report);
  return report;
}

function indexFinalJsonByRole(jsonTree) {
  const byRole = new Map();
  function walk(node) {
    if (!node || typeof node !== "object") return;
    const role = semanticRoleName(node);
    if (role && !byRole.has(role)) byRole.set(role, node);
    const kids = Array.isArray(node.children) ? node.children : [];
    for (const child of kids) walk(child);
  }
  walk(jsonTree);
  return byRole;
}

function countTextLines(characters) {
  if (characters == null) return 1;
  const parts = String(characters).split(/\r\n|\n|\u2028|\u2029|\u000b|\u000c|\u0085/);
  return Math.max(1, parts.length);
}

function estimateWrappedTextLines(characters, fontSize, boxWidth, role) {
  const text = String(characters == null ? "" : characters);
  const explicitLines = text.split(/\r\n|\n|\u2028|\u2029|\u000b|\u000c|\u0085/);
  const explicitLineCount = Math.max(1, explicitLines.length);
  const longest = explicitLines.reduce((max, line) => Math.max(max, line.length), 0);
  if (!longest || !fontSize || !boxWidth) return explicitLineCount;
  const avgCharWidth = role === "headline" ? 0.54 : 0.48;
  const estimated = Math.ceil((longest * fontSize * avgCharWidth) / Math.max(1, boxWidth));
  return Math.max(explicitLineCount, estimated);
}

function singleLineText(value) {
  return String(value == null ? "" : value).replace(/[\r\n\u2028\u2029\u000b\u000c\u0085]+/g, "");
}

function estimatedSingleLineTextWidth(node) {
  const chars = String(node && node.characters ? node.characters : "");
  const fontSize = typeof node.fontSize === "number" && Number.isFinite(node.fontSize) ? node.fontSize : 16;
  const letterSpacing =
    node &&
    node.letterSpacing &&
    typeof node.letterSpacing === "object" &&
    node.letterSpacing.unit === "PIXELS" &&
    typeof node.letterSpacing.value === "number"
      ? node.letterSpacing.value
      : 0;
  return Math.max(1, chars.length * fontSize * 0.82 + Math.max(0, chars.length - 1) * letterSpacing + fontSize * 0.35);
}

function estimatedSingleLineTextHeight(node) {
  const fontSize = typeof node.fontSize === "number" && Number.isFinite(node.fontSize) ? node.fontSize : 16;
  const lineHeight =
    node &&
    node.lineHeight &&
    typeof node.lineHeight === "object" &&
    node.lineHeight.unit === "PERCENT" &&
    typeof node.lineHeight.value === "number"
      ? Math.max(1, node.lineHeight.value / 100)
      : 1.05;
  return Math.max(1, fontSize * lineHeight * 1.08);
}

function isAgeBadgeTextRole(role, parentRole) {
  return (
    role === "age_badge" ||
    parentRole === "age_badge" ||
    parentRole === "age_badge_group" ||
    parentRole === "badge_group"
  );
}

function ensureAgeBadgeSingleLine(node, isAgeBadgeText) {
  if (!isAgeBadgeText || !node || !("characters" in node)) return false;
  let changed = false;
  const normalized = singleLineText(node.characters);
  if (normalized !== node.characters) {
    try {
      node.characters = normalized;
      changed = true;
    } catch (e) {
      console.warn("[AGE_BADGE_SINGLE_LINE_CHARACTERS_FAILED]", e);
    }
  }
  try {
    if ("textAutoResize" in node) {
      node.textAutoResize = "WIDTH_AND_HEIGHT";
      changed = true;
    }
  } catch (e) {
    console.warn("[AGE_BADGE_AUTO_RESIZE_FAILED]", e);
  }
  const minWidth = estimatedSingleLineTextWidth(node);
  const minHeight = estimatedSingleLineTextHeight(node);
  if (
    (typeof node.width === "number" && node.width < minWidth) ||
    (typeof node.height === "number" && node.height < minHeight)
  ) {
    resizeNodeIfPossible(
      node,
      Math.max(minWidth, typeof node.width === "number" ? node.width : 1),
      Math.max(minHeight, typeof node.height === "number" ? node.height : 1),
    );
    changed = true;
  }
  return changed;
}

function getSourceTextFontSize(sourceFrame, role) {
  const node = findTextNodeBySemanticRole(sourceFrame, role);
  if (!node) return null;
  try {
    return typeof node.fontSize === "number" && Number.isFinite(node.fontSize) ? node.fontSize : null;
  } catch (_e) {
    return null;
  }
}

function lineHeightMultiplier(item) {
  const lh = item && item.lineHeight;
  if (lh && typeof lh === "object" && lh.unit === "PERCENT" && typeof lh.value === "number") {
    return lh.value / 100;
  }
  return 0.9;
}

function childBoundsLookCanvasAbsolute(parent, child) {
  if (!parent || !child || !parent.bounds || !child.bounds) return false;
  const pb = parent.bounds;
  const cb = child.bounds;
  if (typeof cb.y !== "number" || typeof pb.y !== "number") return false;
  const relY = cb.y - pb.y;
  return relY > (pb.height || 0) + 1;
}

function normalizeAbsoluteChildBoundsInJson(root) {
  let fixed = 0;
  function walk(node) {
    if (!node || typeof node !== "object") return;
    const kids = Array.isArray(node.children) ? node.children : [];
    for (const child of kids) {
      if (childBoundsLookCanvasAbsolute(node, child) && child.bounds) {
        const pb = node.bounds;
        const cb = child.bounds;
        child.bounds = {
          x: (cb.x || 0) - (pb.x || 0),
          y: (cb.y || 0) - (pb.y || 0),
          width: cb.width,
          height: cb.height,
        };
        fixed++;
      }
      walk(child);
    }
  }
  walk(root);
  return { fixed };
}

function getRootJsonBounds(jsonTree) {
  const b = jsonTree && jsonTree.bounds && typeof jsonTree.bounds === "object" ? jsonTree.bounds : {};
  return {
    x: typeof b.x === "number" ? b.x : 0,
    y: typeof b.y === "number" ? b.y : 0,
    width: typeof b.width === "number" ? Math.max(1, b.width) : 1,
    height: typeof b.height === "number" ? Math.max(1, b.height) : 1,
  };
}

function boxesIntersect(a, b) {
  if (!a || !b) return false;
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function keepLargeBoundsPartiallyVisible(bounds, canvas, visibleRatio) {
  if (!bounds || !canvas) return false;
  const ratio = Math.min(0.9, Math.max(0.1, visibleRatio || 0.4));
  const minX = canvas.x - bounds.width * (1 - ratio);
  const maxX = canvas.x + canvas.width - bounds.width * ratio;
  const minY = canvas.y - bounds.height * (1 - ratio);
  const maxY = canvas.y + canvas.height - bounds.height * ratio;
  const nextX = Number(clampNumber(bounds.x || 0, minX, maxX).toFixed(2));
  const nextY = Number(clampNumber(bounds.y || 0, minY, maxY).toFixed(2));
  const changed = Math.abs((bounds.x || 0) - nextX) > 0.01 || Math.abs((bounds.y || 0) - nextY) > 0.01;
  bounds.x = nextX;
  bounds.y = nextY;
  return changed;
}

function moveBoundsTo(bounds, x, y) {
  if (!bounds) return false;
  const nextX = Number((Number(x) || 0).toFixed(2));
  const nextY = Number((Number(y) || 0).toFixed(2));
  const changed = Math.abs((bounds.x || 0) - nextX) > 0.01 || Math.abs((bounds.y || 0) - nextY) > 0.01;
  bounds.x = nextX;
  bounds.y = nextY;
  return changed;
}

function intersectionBottomRight(a, b) {
  if (!a || !b) return null;
  const right = Math.min((a.x || 0) + (a.width || 0), (b.x || 0) + (b.width || 0));
  const bottom = Math.min((a.y || 0) + (a.height || 0), (b.y || 0) + (b.height || 0));
  const left = Math.max(a.x || 0, b.x || 0);
  const top = Math.max(a.y || 0, b.y || 0);
  if (right <= left || bottom <= top) return null;
  return { x: right, y: bottom };
}

function ensureGradientStrongStopAtEnd(node) {
  if (!node || !Array.isArray(node.fills)) return false;
  let changed = false;
  for (const fill of node.fills) {
    if (!fill || String(fill.type || "").indexOf("GRADIENT_") !== 0 || !Array.isArray(fill.gradientStops)) continue;
    let strongest = null;
    for (const stop of fill.gradientStops) {
      const alpha =
        stop && stop.color && typeof stop.color.a === "number" ? stop.color.a : 1;
      if (!strongest || alpha > strongest.alpha) {
        strongest = { stop, alpha };
      }
    }
    if (strongest && typeof strongest.stop.position === "number" && strongest.stop.position < 0.5) {
      fill.gradientStops = fill.gradientStops
        .map((stop) => Object.assign({}, stop, { position: Number((1 - (Number(stop.position) || 0)).toFixed(6)) }))
        .sort((a, b) => (a.position || 0) - (b.position || 0));
      changed = true;
    }
  }
  return changed;
}

function hideBackgroundGradientJsonNodes(jsonTree) {
  let hidden = 0;
  function walk(node) {
    if (!node || typeof node !== "object") return;
    if (/^background_gradient_/.test(semanticRoleName(node))) {
      if (node.visible !== false || node.opacity !== 0) hidden++;
      node.visible = false;
      node.opacity = 0;
    }
    const kids = Array.isArray(node.children) ? node.children : [];
    for (const child of kids) walk(child);
  }
  walk(jsonTree);
  return { hidden };
}

function polishDecorativeJsonBounds(finalJson, sourceFrame) {
  if (!finalJson || typeof finalJson !== "object") {
    return { backendGradientsPreserved: false, gradientsHidden: 0, gradientsMoved: 0, starsMoved: 0 };
  }
  const backendGradientsPreserved = sourceFrameHasGradient(sourceFrame);
  const gradientHideReport = backendGradientsPreserved
    ? { hidden: 0 }
    : hideBackgroundGradientJsonNodes(finalJson);
  const byRole = indexFinalJsonByRole(finalJson);
  const canvas = getRootJsonBounds(finalJson);
  const background = byRole.get("background_shape");
  const backgroundBounds = background && background.bounds && typeof background.bounds === "object" ? background.bounds : null;
  const kids = Array.isArray(finalJson.children) ? finalJson.children : [];
  let gradientsMoved = 0;
  let starsMoved = 0;

  const anchor = backgroundBounds || canvas;
  const gradients = kids
    .filter((node) => /^background_gradient_/.test(semanticRoleName(node)) && node.bounds)
    .sort((a, b) => semanticRoleName(a).localeCompare(semanticRoleName(b)));
  if (!backendGradientsPreserved) {
    if (gradients[0] && gradients[0].bounds) {
      const b = gradients[0].bounds;
      const moved = moveBoundsTo(
        b,
        anchor.x - b.width * 0.04,
        anchor.y - b.height * 0.01,
      );
      const clamped = keepLargeBoundsPartiallyVisible(b, canvas, 0.48);
      if (moved || clamped) gradientsMoved++;
    }
    if (gradients[1] && gradients[1].bounds) {
      const b = gradients[1].bounds;
      const corner = intersectionBottomRight(canvas, anchor) || {
        x: canvas.x + canvas.width,
        y: canvas.y + canvas.height,
      };
      const moved = moveBoundsTo(
        b,
        corner.x - b.width,
        corner.y - b.height,
      );
      const flipped = ensureGradientStrongStopAtEnd(gradients[1]);
      if (moved || flipped) gradientsMoved++;
    }
    for (let gi = 2; gi < gradients.length; gi++) {
      const b = gradients[gi].bounds;
      if (b && !boxesIntersect(b, canvas)) {
        const moved = moveBoundsTo(
          b,
          anchor.x + anchor.width - b.width * 0.5,
          anchor.y + anchor.height - b.height * 0.78,
        );
        const clamped = keepLargeBoundsPartiallyVisible(b, canvas, 0.35);
        if (moved || clamped) gradientsMoved++;
      }
    }
  }

  if (backgroundBounds) {
    const stars = kids
      .filter((node) => /^star_decoration_/.test(semanticRoleName(node)) && node.bounds)
      .sort((a, b) => (b.bounds.width * b.bounds.height) - (a.bounds.width * a.bounds.height));
    const upperStar = stars[0];
    if (upperStar && upperStar.bounds) {
      const b = upperStar.bounds;
      const topY = backgroundBounds.y + backgroundBounds.height * 0.07;
      const leftX = Math.max(canvas.x + canvas.width * 0.02, backgroundBounds.x + backgroundBounds.width * 0.03);
      if (Math.abs((b.y || 0) - topY) > 0.5 || Math.abs((b.x || 0) - leftX) > 0.5) {
        b.x = Number(
          clampNumber(leftX, canvas.x - b.width * 0.25, canvas.x + canvas.width - b.width * 0.35).toFixed(2),
        );
        b.y = Number(clampNumber(topY, canvas.y, canvas.y + canvas.height - b.height * 0.35).toFixed(2));
        starsMoved++;
      }
    }
  }

  return {
    backendGradientsPreserved,
    gradientsHidden: gradientHideReport.hidden,
    gradientsMoved,
    starsMoved,
  };
}

function estimateHeadlineFontSize(headlineItem, boxWidth, boxHeight, lineCount, maxFont) {
  const lines = Math.max(1, lineCount);
  const lh = lineHeightMultiplier(headlineItem);
  const byHeight = ((boxHeight || 200) / lines) * lh * 0.98;
  const chars = String((headlineItem && headlineItem.characters) || "");
  const longest = chars
    .split(/\r\n|\n|\u2028|\u2029/)
    .reduce((max, line) => Math.max(max, line.length), 0);
  const byWidth = longest > 0 ? (boxWidth / longest) * 1.55 : maxFont;
  return Math.max(12, Math.min(maxFont, byHeight, byWidth));
}

/**
 * Portrait-only JSON pass: stack brand → headline → subheadline while preserving backend headline font size,
 * banner-absolute text bounds, TOP vertical alignment (avoids BOTTOM bleed into brand).
 */
function polishPortraitTypographyInJson(finalJson, sourceFrame, targetResolution) {
  const targetW = Number(targetResolution && targetResolution.width);
  const targetH = Number(targetResolution && targetResolution.height);
  if (!finalJson || !targetW || !targetH || targetW >= targetH) {
    return { applied: false, reason: "not_portrait" };
  }

  const normalizeReport = normalizeAbsoluteChildBoundsInJson(finalJson);
  const byRole = indexFinalJsonByRole(finalJson);
  const brand = byRole.get("brand_group");
  const headlineGroup = byRole.get("headline_group");
  const headline = byRole.get("headline");
  const sub = byRole.get("subheadline_delivery_time");
  if (!headline || !headline.bounds) {
    return { applied: false, reason: "no_headline", normalized: normalizeReport.fixed };
  }

  const jsonFont = typeof headline.fontSize === "number" ? headline.fontSize : 72;

  const marginX = Math.round(targetW * 0.05);
  const contentTop = Math.round(targetH * 0.52);
  const stackGap = Math.round(targetH * 0.022);
  const innerPad = Math.round(targetH * 0.012);

  if (brand && brand.bounds) {
    brand.bounds.x = marginX;
    brand.bounds.width = Math.max(1, targetW - 2 * marginX);
    if (typeof brand.bounds.y !== "number" || brand.bounds.y < contentTop - 4) {
      brand.bounds.y = contentTop;
    }
  }

  const brandBottom =
    brand && brand.bounds && typeof brand.bounds.y === "number" && typeof brand.bounds.height === "number"
      ? brand.bounds.y + brand.bounds.height
      : contentTop + Math.round(targetH * 0.08);

  const boxW = Math.max(1, targetW - 2 * marginX);
  const fontSize = Math.round(jsonFont * 10) / 10;
  const textBoxW = Math.max(1, boxW - innerPad * 2);
  const headlineLines = estimateWrappedTextLines(headline.characters || "", fontSize, textBoxW, "headline");

  const lhMul = lineHeightMultiplier(headline);
  const headlineLineH = fontSize * Math.max(0.96, lhMul) * 1.04;
  const headlineBoxH = Math.ceil(headlineLineH * headlineLines);
  const groupY = brandBottom + stackGap;

  headline.fontSize = fontSize;
  headline.textAlignHorizontal = "CENTER";
  headline.textAlignVertical = "TOP";
  headline.textAutoResize = "NONE";

  let subBoxH = 0;
  let subGap = 0;
  if (sub && sub.bounds) {
    subGap = Math.max(6, Math.round(targetH * 0.006));
    const subFontCap = Math.min(
      typeof sub.fontSize === "number" ? sub.fontSize : 40,
      fontSize * 0.58,
      targetW * 0.052,
    );
    sub.fontSize = Math.round(subFontCap * 10) / 10;
    sub.textAlignHorizontal = "CENTER";
    sub.textAlignVertical = "TOP";
    sub.textAutoResize = "NONE";
    subBoxH = Math.ceil(sub.fontSize * Math.max(1.02, lineHeightMultiplier(sub)) * 1.16);
  }

  const groupH = innerPad + headlineBoxH + (sub ? subGap + subBoxH : 0) + innerPad;
  const bottomMargin = Math.round(targetH * 0.04);
  const groupBottom = groupY + groupH;
  let finalGroupY = groupY;
  if (groupBottom > targetH - bottomMargin && headlineGroup && headlineGroup.bounds) {
    const shift = groupBottom - (targetH - bottomMargin);
    finalGroupY = Math.max(contentTop, groupY - shift);
    if (brand && brand.bounds) brand.bounds.y = Math.max(contentTop, (brand.bounds.y || contentTop) - shift);
  }
  const groupX = marginX;

  if (headlineGroup && headlineGroup.bounds) {
    headlineGroup.bounds.x = groupX;
    headlineGroup.bounds.y = finalGroupY;
    headlineGroup.bounds.width = boxW;
    headlineGroup.bounds.height = groupH;
  }

  headline.bounds = {
    x: groupX + innerPad,
    y: finalGroupY + innerPad,
    width: Math.max(1, boxW - innerPad * 2),
    height: headlineBoxH,
  };

  if (sub && sub.bounds) {
    sub.bounds = {
      x: groupX + innerPad,
      y: finalGroupY + innerPad + headlineBoxH + subGap,
      width: Math.max(1, boxW - innerPad * 2),
      height: subBoxH,
    };
  }

  const decorationReport = polishDecorativeJsonBounds(finalJson, sourceFrame);

  console.log("[PORTRAIT_TYPO_POLISH]", {
    fontSize,
    source: "backend_final_json",
    headlineLines,
    groupY: headlineGroup && headlineGroup.bounds ? headlineGroup.bounds.y : null,
    groupH,
    normalized: normalizeReport.fixed,
    decorations: decorationReport,
  });

  return {
    applied: true,
    fontSize,
    normalized: normalizeReport.fixed,
    decorations: decorationReport,
    groupY: headlineGroup && headlineGroup.bounds ? headlineGroup.bounds.y : null,
  };
}

async function fitFigmaTextNodeToBox(node, maxWidth, maxHeight, opts) {
  if (!node || !("characters" in node)) return { fitted: false };
  const role = opts && opts.role ? opts.role : node.name;
  const minFont = opts && typeof opts.minFont === "number" ? opts.minFont : 12;
  let steps = 0;
  const maxSteps = 48;
  try {
    if ("textAutoResize" in node) node.textAutoResize = "NONE";
  } catch (_e) {
    /* best-effort */
  }
  if (typeof maxWidth === "number" && typeof maxHeight === "number") {
    resizeNodeIfPossible(node, maxWidth, maxHeight);
  }
  await loadAllFontsForTextNode(node, role);
  if (opts && opts.alignVertical === "TOP") {
    try {
      node.textAlignVertical = "TOP";
    } catch (_e) {
      /* best-effort */
    }
  }
  let targetFont =
    opts && typeof opts.maxFont === "number" ? opts.maxFont : typeof node.fontSize === "number" ? node.fontSize : 80;
  try {
    node.fontSize = Math.max(minFont, targetFont);
    applyUniformTextRangeValue(node, "setRangeFontSize", "FONT_SIZE", Math.max(minFont, targetFont), role);
  } catch (_e) {
    /* best-effort */
  }
  while (steps < maxSteps && node.fontSize > minFont) {
    const overflowW = typeof maxWidth === "number" && node.width > maxWidth + 0.5;
    const overflowH = typeof maxHeight === "number" && node.height > maxHeight + 0.5;
    if (!overflowW && !overflowH) break;
    targetFont = Math.max(minFont, node.fontSize - 1);
    try {
      node.fontSize = targetFont;
      applyUniformTextRangeValue(node, "setRangeFontSize", "FONT_SIZE", targetFont, role);
    } catch (e) {
      console.warn("[TEXT_FIT_FONT_FAILED]", role, e);
      break;
    }
    steps++;
  }
  return { fitted: steps > 0, fontSize: node.fontSize, steps };
}

function applyExactTextFontSize(node, fontSize, role) {
  if (!node || !("characters" in node) || typeof fontSize !== "number" || !Number.isFinite(fontSize)) {
    return false;
  }
  const size = Math.max(1, fontSize);
  try {
    node.fontSize = size;
    applyUniformTextRangeValue(node, "setRangeFontSize", "FONT_SIZE", size, role);
    return true;
  } catch (e) {
    console.warn("[TEXT_EXACT_FONT_SIZE_APPLY_FAILED]", role, fontSize, e);
    return false;
  }
}

async function polishPortraitTypographyOnFrame(convertedFrame, finalJson, targetResolution, sourceContentMap) {
  const targetW = Number(targetResolution && targetResolution.width);
  const targetH = Number(targetResolution && targetResolution.height);
  if (!convertedFrame || !targetW || !targetH || targetW >= targetH) {
    return { applied: false, reason: "not_portrait" };
  }

  const byRole = indexFinalJsonByRole(finalJson);
  const headlineItem = byRole.get("headline");
  const headlineGroupItem = byRole.get("headline_group");
  const headlineNode = findTextNodeBySemanticRole(convertedFrame, "headline");
  if (!headlineItem || !headlineNode) {
    return { applied: false, reason: "no_headline_node" };
  }

  function localTextPointForJsonBounds(node, bounds) {
    if (!node || !bounds) return { x: node ? node.x : 0, y: node ? node.y : 0 };
    const parentIsRoot = !node.parent || node.parent.id === convertedFrame.id;
    const parentBounds =
      !parentIsRoot && headlineGroupItem && headlineGroupItem.bounds && typeof headlineGroupItem.bounds === "object"
        ? headlineGroupItem.bounds
        : null;
    return {
      x: typeof bounds.x === "number" ? bounds.x - (parentBounds && typeof parentBounds.x === "number" ? parentBounds.x : 0) : node.x,
      y: typeof bounds.y === "number" ? bounds.y - (parentBounds && typeof parentBounds.y === "number" ? parentBounds.y : 0) : node.y,
    };
  }

  const b = headlineItem.bounds;
  if (b) {
    const p = localTextPointForJsonBounds(headlineNode, b);
    headlineNode.x = p.x;
    headlineNode.y = p.y;
    if (typeof b.width === "number" && typeof b.height === "number") {
      resizeNodeIfPossible(headlineNode, b.width, b.height);
    }
  }

  const backendHeadlineFontSize =
    typeof headlineItem.fontSize === "number" && Number.isFinite(headlineItem.fontSize)
      ? headlineItem.fontSize
      : headlineNode.fontSize || 72;
  const fontSizeApplied = applyExactTextFontSize(headlineNode, backendHeadlineFontSize, "headline");
  try {
    headlineNode.textAlignVertical = "TOP";
    headlineNode.textAlignHorizontal = textAlignHorizontalForTarget(headlineItem, sourceContentMap) || "CENTER";
  } catch (_e) {
    /* best-effort */
  }

  const subItem = byRole.get("subheadline_delivery_time");
  const subNode = findTextNodeBySemanticRole(convertedFrame, "subheadline_delivery_time");
  if (subItem && subNode && subItem.bounds) {
    const sb = subItem.bounds;
    const p = localTextPointForJsonBounds(subNode, sb);
    subNode.x = p.x;
    subNode.y = p.y;
    if (typeof sb.width === "number" && typeof sb.height === "number") {
      resizeNodeIfPossible(subNode, sb.width, sb.height);
    }
    await fitFigmaTextNodeToBox(subNode, sb.width, sb.height, {
      role: "subheadline_delivery_time",
      maxFont: subItem.fontSize || subNode.fontSize,
      alignVertical: "TOP",
    });
    try {
      subNode.textAlignVertical = "TOP";
      subNode.textAlignHorizontal = textAlignHorizontalForTarget(subItem, sourceContentMap) || "CENTER";
    } catch (_e) {
      /* best-effort */
    }
  }

  return {
    applied: true,
    headline: {
      fitted: false,
      fontSize: headlineNode.fontSize,
      backendFontSize: backendHeadlineFontSize,
      exactBackendFontSizeApplied: fontSizeApplied,
    },
  };
}

function figmaNodeAllowedInSource(node, sourceIndex, cloneRoot) {
  if (!node || !sourceIndex || (cloneRoot && node.id === cloneRoot.id)) return true;
  let storedId = "";
  let semanticName = "";
  try {
    storedId = String(node.getPluginData("originalNodeId") || "").trim();
    semanticName = String(node.getPluginData("semanticName") || node.getPluginData("semanticRole") || "").trim();
  } catch (_e) {
    storedId = "";
    semanticName = "";
  }
  const nodeId = String(node.id || "").trim();
  const role = String(node.name || "").trim() || semanticName;
  if ((storedId && sourceIndex.ids.has(storedId)) || (nodeId && sourceIndex.ids.has(nodeId))) return true;
  if (role && sourceIndex.roles.has(role)) return true;
  if (LOGO_PART_ROLES.has(role) && sourceIndex.roles.has("logo")) return true;
  return false;
}

/**
 * Final safety pass: remove any Figma node that was not present on the source banner.
 */
function pruneFigmaNodesNotInSource(sourceFrame, cloneRoot) {
  const sourceIndex = collectSourceElementIndex(sourceFrame);
  let removed = 0;

  function visitPost(node) {
    if (!node || node === cloneRoot || !("children" in node) || !Array.isArray(node.children)) return;
    const kids = [...node.children];
    for (const child of kids) visitPost(child);
    const kids2 = [...node.children];
    for (const child of kids2) {
      if (figmaNodeAllowedInSource(child, sourceIndex, cloneRoot)) continue;
      try {
        child.remove();
        removed++;
        console.log("[SOURCE_ELEMENT_PRUNE]", child.name, child.id);
      } catch (e) {
        console.warn("[SOURCE_ELEMENT_PRUNE_FAILED]", child && child.name, e);
      }
    }
  }

  visitPost(cloneRoot);
  return { removed };
}

function attachTargetOrientationToSourceContentMap(sourceContentMap, targetResolution) {
  if (!sourceContentMap || !targetResolution) return sourceContentMap;
  sourceContentMap.targetIsLandscape = Number(targetResolution.width) > Number(targetResolution.height);
  return sourceContentMap;
}

function textAlignHorizontalForTarget(item, sourceContentMap) {
  if (sourceContentMap && typeof sourceContentMap.targetIsLandscape === "boolean") {
    return sourceContentMap.targetIsLandscape ? "LEFT" : "CENTER";
  }
  return item && item.textAlignHorizontal ? item.textAlignHorizontal : null;
}

function textAlignVerticalForTarget(item, sourceContentMap) {
  if (
    sourceContentMap &&
    sourceContentMap.targetIsLandscape === false &&
    semanticRoleName(item) === "headline"
  ) {
    return "TOP";
  }
  return item && item.textAlignVertical ? item.textAlignVertical : null;
}

function sourceContentLookupKeys(item) {
  return [
    item && item.id,
    item && item.source_figma_id,
    item && item.sourceFigmaId,
    item && item.figma_node_id,
    item && item.node_id,
  ];
}

function resolveSourceCharactersForJsonText(item, sourceContentMap) {
  if (!item || !sourceContentMap) return item && "characters" in item ? item.characters : undefined;
  const ids = [
    ...sourceContentLookupKeys(item),
  ];
  for (const id of ids) {
    const key = String(id || "").trim();
    if (key && sourceContentMap.textById && sourceContentMap.textById.has(key)) return sourceContentMap.textById.get(key);
    if (key && sourceContentMap.byId && sourceContentMap.byId.has(key)) return sourceContentMap.byId.get(key);
  }
  const path = String(item.path || "").trim();
  if (path && sourceContentMap.textByPath && sourceContentMap.textByPath.has(path)) return sourceContentMap.textByPath.get(path);
  if (path && sourceContentMap.byPath && sourceContentMap.byPath.has(path)) return sourceContentMap.byPath.get(path);

  const roleKeys = [item.name, item.role, item.semantic_name, item.semanticName];
  for (const role of roleKeys) {
    const key = String(role || "").trim();
    if (key && sourceContentMap.textByRole && sourceContentMap.textByRole.has(key)) return sourceContentMap.textByRole.get(key);
    if (key && sourceContentMap.byRole && sourceContentMap.byRole.has(key)) return sourceContentMap.byRole.get(key);
  }
  return "characters" in item ? item.characters : undefined;
}

function resolveSourceImageFillsForJsonNode(item, sourceContentMap) {
  if (!item || !sourceContentMap) return null;
  for (const id of sourceContentLookupKeys(item)) {
    const key = String(id || "").trim();
    if (key && sourceContentMap.imageFillsById && sourceContentMap.imageFillsById.has(key)) {
      return sourceContentMap.imageFillsById.get(key);
    }
  }
  const path = String(item.path || "").trim();
  if (path && sourceContentMap.imageFillsByPath && sourceContentMap.imageFillsByPath.has(path)) {
    return sourceContentMap.imageFillsByPath.get(path);
  }
  const roleKeys = [item.name, item.role, item.semantic_name, item.semanticName];
  for (const role of roleKeys) {
    const key = String(role || "").trim();
    if (key && sourceContentMap.imageFillsByRole && sourceContentMap.imageFillsByRole.has(key)) {
      return sourceContentMap.imageFillsByRole.get(key);
    }
  }
  return null;
}

function applySourceImageContent(node, item, sourceContentMap) {
  if (!node || !item || "characters" in node) return 0;
  const sourceImageFills = resolveSourceImageFillsForJsonNode(item, sourceContentMap);
  if (!sourceImageFills || !("fills" in node)) return 0;
  try {
    node.fills = sourceImageFills;
    return 1;
  } catch (e) {
    console.warn("[IMAGE_CONTENT_APPLY_FAILED]", item && (item.name || item.id || item.path), e);
    return 0;
  }
}

function resolveSourceVisualNodeForJsonNode(item, sourceContentMap) {
  if (!item || !sourceContentMap) return null;
  for (const id of sourceContentLookupKeys(item)) {
    const key = String(id || "").trim();
    if (key && sourceContentMap.visualNodeById && sourceContentMap.visualNodeById.has(key)) {
      return sourceContentMap.visualNodeById.get(key);
    }
  }
  const path = String(item.path || "").trim();
  if (path && sourceContentMap.visualNodeByPath && sourceContentMap.visualNodeByPath.has(path)) {
    return sourceContentMap.visualNodeByPath.get(path);
  }
  const roleKeys = [item.name, item.role, item.semantic_name, item.semanticName];
  for (const role of roleKeys) {
    const key = String(role || "").trim();
    if (key && sourceContentMap.visualNodeByRole && sourceContentMap.visualNodeByRole.has(key)) {
      return sourceContentMap.visualNodeByRole.get(key);
    }
  }
  return null;
}

function shouldCloneSourceVisualForJsonItem(item, isRoot) {
  if (!item || typeof item !== "object" || isRoot || "characters" in item) return false;
  const type = normalizeType(item.type || "");
  const name = semanticRoleName(item);
  return (
    type === "vector" ||
    type === "star" ||
    type === "boolean operation" ||
    type === "boolean_operation" ||
    name === "logo" ||
    /^brand_name_/.test(name)
  );
}

function cloneSourceVisualIntoJson(parent, item, parentBounds, sourceContentMap) {
  if (!parent || typeof parent.appendChild !== "function") return null;
  const sourceNode = resolveSourceVisualNodeForJsonNode(item, sourceContentMap);
  if (!sourceNode || typeof sourceNode.clone !== "function") return null;
  const node = sourceNode.clone();
  node.name = String(item.name || sourceNode.name || "visual_content");
  try {
    if (item.id) node.setPluginData("originalNodeId", String(item.id));
    node.setPluginData("semanticName", String(item.name || ""));
  } catch (_e) {
    /* plugin data best-effort */
  }
  parent.appendChild(node);
  applyJsonClippingBehavior(node, item);
  applyBoundsFromAbsoluteJson(node, item, { bounds: parentBounds });
  applyJsonVisualStyle(node, item);
  return node;
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

function buildSemanticCloneFrameTitle(sourceFrameName, semanticRootName) {
  const src = sanitizeLayerName(String(sourceFrameName || "").trim()) || "frame";
  const sem = sanitizeLayerName(String(semanticRootName || "").trim()) || "semantic";
  return `${src} · ${sem}`;
}

/**
 * Duplicate a frame to the right of the source (same Y). Caller should stamp ``originalNodeId`` on the
 * source tree before cloning so the copy can be matched to backend JSON ``id`` fields.
 */
function cloneFrameBesideSource(sourceFrame) {
  const clone = sourceFrame.clone();
  clone.x = sourceFrame.x + sourceFrame.width + BESIDE_FRAME_GAP;
  clone.y = sourceFrame.y;
  figma.currentPage.appendChild(clone);
  return clone;
}

function isStrictDescendantOf(node, ancestorCandidate) {
  if (!node || !ancestorCandidate) return false;
  let p = node.parent;
  while (p) {
    if (p.id === ancestorCandidate.id) return true;
    p = p.parent;
  }
  return false;
}

/**
 * Reparent and reorder nodes in ``cloneRoot`` so parent/child order matches ``jsonTree.children``
 * (same serialized Figma ``id`` keys as ``collectClonedNodesByOriginalId``).
 * Call before ``finalizeSemanticLayerNamesFromJson`` / ``applyJsonTreeNamesByOriginalIds`` so the layer list matches backend hierarchy.
 */
function syncCloneHierarchyToJsonTree(jsonTree, cloneRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
  let reparentMoves = 0;
  const errors = [];

  function resolveParentNode(jsonItem) {
    const pid = String(jsonItem.id || "").trim();
    if (pid && byId.has(pid)) return byId.get(pid);
    if (jsonItem === jsonTree) return cloneRoot;
    return null;
  }

  function sync(jsonItem) {
    if (!jsonItem || typeof jsonItem !== "object") return;
    const parentNode = resolveParentNode(jsonItem);
    const kids = Array.isArray(jsonItem.children) ? jsonItem.children : [];

    if (parentNode && typeof parentNode.insertChild === "function") {
      let slot = 0;
      for (let i = 0; i < kids.length; i++) {
        const cj = kids[i];
        if (!cj || typeof cj !== "object") continue;
        const cid = String(cj.id || "").trim();
        if (!cid) continue;
        const childNode = byId.get(cid);
        if (!childNode || childNode.id === cloneRoot.id) continue;
        if (isStrictDescendantOf(parentNode, childNode)) {
          console.warn("syncCloneHierarchy: skip (would cycle)", cid, "under", jsonItem.id);
          continue;
        }
        const already =
          childNode.parent &&
          childNode.parent.id === parentNode.id &&
          parentNode.children.indexOf(childNode) === slot;
        if (already) {
          slot++;
          continue;
        }
        try {
          parentNode.insertChild(slot, childNode);
          reparentMoves++;
        } catch (e) {
          const msg = e && e.message ? e.message : String(e);
          errors.push(cid + ":" + msg);
          console.warn("syncCloneHierarchy: insertChild failed", cid, e);
          continue;
        }
        slot++;
      }
    }

    for (let j = 0; j < kids.length; j++) {
      if (kids[j] && typeof kids[j] === "object") sync(kids[j]);
    }
  }

  sync(jsonTree);
  return { reparentMoves, errors };
}

/**
 * Collect every concrete ``id`` in a semantic / ``final_json`` tree (for pruning clones).
 */
function collectFinalJsonNodeIds(jsonTree) {
  const ids = new Set();
  function walk(item) {
    if (!item || typeof item !== "object") return;
    const id = String(item.id || "").trim();
    if (id) ids.add(id);
    const kids = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < kids.length; i++) walk(kids[i]);
  }
  walk(jsonTree);
  return ids;
}

/**
 * Re-stamp ``originalNodeId`` on a Figma subtree from a JSON subtree (index-aligned children).
 */
function stampCloneSubtreeOriginalIdsFromJson(jRoot, fRoot) {
  let stamped = 0;
  let mismatchWarn = 0;

  function walk(jNode, fNode) {
    if (!jNode || typeof jNode !== "object" || !fNode) return;
    const jid = String(jNode.id || "").trim();
    if (jid) {
      try {
        const existing = String(fNode.getPluginData("originalNodeId") || "").trim();
        if (!existing) {
          fNode.setPluginData("originalNodeId", jid);
          stamped++;
        } else if (existing !== jid) {
          mismatchWarn++;
          console.warn("stampCloneIds: preserve existing originalNodeId", {
            figmaId: fNode.id,
            existing,
            wanted: jid,
            jsonName: jNode.name,
            figmaName: fNode.name,
          });
        }
      } catch (e) {
        console.warn("stampCloneIds: setPluginData failed", jid, e);
      }
    }
    const jch = Array.isArray(jNode.children) ? jNode.children : [];
    if (!("children" in fNode) || !Array.isArray(fNode.children)) return;
    const fch = fNode.children;
    if (jch.length !== fch.length) {
      mismatchWarn++;
      console.warn(
        "stampCloneIds: child count mismatch at",
        jid || "root",
        "json=",
        jch.length,
        "figma=",
        fch.length,
      );
    }
    const n = Math.min(jch.length, fch.length);
    for (let i = 0; i < n; i++) walk(jch[i], fch[i]);
  }

  walk(jRoot, fRoot);
  return { stamped, mismatchWarn };
}

/**
 * Re-stamp ``originalNodeId`` on the clone from ``jsonTree`` using **index-aligned** pairing:
 * ``jsonTree.children[i]`` ↔ ``cloneRoot.children[i]`` at every level.
 * Call after hierarchy is stable; often invoked **twice** in ``applyFinalJsonCloneReconstruction`` (initial
 * pairing on the raw clone, then again after prune / stray cleanup).
 */
function stampCloneOriginalIdsFromJson(jsonTree, cloneRoot) {
  return stampCloneSubtreeOriginalIdsFromJson(jsonTree, cloneRoot);
}

/**
 * Figma ignores ``node.name`` (and plugin data on inner nodes) inside INSTANCE subtrees until detached.
 * Detach every INSTANCE that still appears in ``jsonTree`` with semantic ``children`` so later passes
 * can sync hierarchy and apply ``final_json`` / ``semantic_json`` names.
 */
function detachInstancesForFinalJson(jsonTree, cloneRoot) {
  let detached = 0;
  const post = [];

  function collectPost(j) {
    if (!j || typeof j !== "object") return;
    const ch = Array.isArray(j.children) ? j.children : [];
    for (let i = 0; i < ch.length; i++) collectPost(ch[i]);
    post.push(j);
  }
  collectPost(jsonTree);

  for (let pi = 0; pi < post.length; pi++) {
    const jNode = post[pi];
    const jid = String(jNode.id || "").trim();
    const jkids = Array.isArray(jNode.children) ? jNode.children : [];
    if (!jid || jkids.length === 0) continue;

    const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
    const fig = byId.get(jid);
    if (!fig || fig.type !== "INSTANCE") continue;

    let detachedFrame = null;
    try {
      detachedFrame = fig.detachInstance();
      detached++;
    } catch (e) {
      console.warn("detachInstancesForFinalJson: detachInstance failed", jid, e);
      continue;
    }

    const sem = String(jNode.name || "").trim();
    if (sem) {
      try {
        setSemanticName(detachedFrame, sem);
      } catch (e2) {
        console.warn("detachInstancesForFinalJson: rename detached root failed", jid, e2);
      }
    }

    stampCloneSubtreeOriginalIdsFromJson(jNode, detachedFrame);
  }

  return { detached };
}

/**
 * For every JSON parent, reorder Figma children so indices match ``final_json.children`` order
 * (same serialized ids as ``collectClonedNodesByOriginalId``).
 */
function reorderCloneChildrenPerFinalJson(jsonTree, cloneRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
  let moves = 0;

  function walk(jParent) {
    if (!jParent || typeof jParent !== "object") return;
    const jkids = Array.isArray(jParent.children) ? jParent.children : [];
    for (let i = 0; i < jkids.length; i++) walk(jkids[i]);

    let parentFigma = null;
    if (jParent === jsonTree) {
      parentFigma = cloneRoot;
    } else {
      const pid = String(jParent.id || "").trim();
      if (pid && byId.has(pid)) parentFigma = byId.get(pid);
    }
    if (!parentFigma || typeof parentFigma.insertChild !== "function" || jkids.length === 0) return;

    for (let target = 0; target < jkids.length; target++) {
      const cid = String(jkids[target].id || "").trim();
      if (!cid) continue;
      const childNode = byId.get(cid);
      if (!childNode || !childNode.parent || childNode.parent.id !== parentFigma.id) continue;
      const cur = parentFigma.children.indexOf(childNode);
      if (cur !== target) {
        try {
          parentFigma.insertChild(target, childNode);
          moves++;
        } catch (e) {
          console.warn("reorderCloneChildren: insertChild failed", cid, e);
        }
      }
    }
  }

  walk(jsonTree);
  return { moves };
}

/**
 * Remove Figma children that are not listed under this JSON parent (stale mid wrappers like the old
 * boolean shell around ``logo_fore``). Keeps only nodes whose ``originalNodeId`` is in the JSON
 * child id set for that parent.
 */
function removeStrayFigmaChildrenNotInJson(jsonTree, cloneRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
  let removed = 0;

  function walk(jParent) {
    if (!jParent || typeof jParent !== "object") return;
    const jkids = Array.isArray(jParent.children) ? jParent.children : [];
    for (let i = 0; i < jkids.length; i++) walk(jkids[i]);

    let parentFigma = null;
    if (jParent === jsonTree) {
      parentFigma = cloneRoot;
    } else {
      const pid = String(jParent.id || "").trim();
      if (pid && byId.has(pid)) parentFigma = byId.get(pid);
    }
    if (!parentFigma || !("children" in parentFigma) || !Array.isArray(parentFigma.children)) return;
    if (jkids.length === 0) return;

    const wanted = new Set();
    for (let w = 0; w < jkids.length; w++) {
      const id = String(jkids[w].id || "").trim();
      if (id) wanted.add(id);
    }
    if (wanted.size === 0) return;

    const toRemove = [];
    for (const c of parentFigma.children) {
      let oid = "";
      try {
        oid = String(c.getPluginData("originalNodeId") || "").trim();
      } catch (_e) {
        oid = "";
      }
      if (!oid || !wanted.has(oid)) toRemove.push(c);
    }
    for (let r = 0; r < toRemove.length; r++) {
      const c = toRemove[r];
      try {
        c.remove();
        removed++;
      } catch (e2) {
        console.warn("removeStray: remove failed", c && c.id, e2);
      }
    }
  }

  walk(jsonTree);
  return { removed };
}

/**
 * Document-space top-left from a node's ``absoluteTransform`` (Figma canvas coordinates).
 */
function getAbsXY(node) {
  if (!node || typeof node.absoluteTransform !== "object" || !node.absoluteTransform) {
    return { x: 0, y: 0 };
  }
  const t = node.absoluteTransform;
  return { x: t[0][2], y: t[1][2] };
}

/**
 * Map a point from **banner / clone-root local** coordinates (same space as ``final_json.bounds``)
 * into **document** coordinates using the clone root frame's current transform.
 */
function jsonBannerPointToDocument(bannerRootFrame, bx, by) {
  const M = bannerRootFrame.absoluteTransform;
  const x = M[0][0] * bx + M[0][1] * by + M[0][2];
  const y = M[1][0] * bx + M[1][1] * by + M[1][2];
  return { x, y };
}

/**
 * Nudge ``node`` so its document top-left matches ``jsonBounds.{x,y}`` in banner space, then apply
 * width/height from JSON. Document delta is converted through the **immediate parent's** inverse
 * linear 2×2 (no special cases for boolean / group / frame).
 */
function forceAbsolutePosition(node, bannerRootFrame, jsonBounds, jsonLabel) {
  if (!node || !jsonBounds || typeof jsonBounds.x !== "number" || typeof jsonBounds.y !== "number") return;
  if (typeof jsonBounds.width === "number" && typeof jsonBounds.height === "number" && "resizeWithoutConstraints" in node) {
    try {
      node.resizeWithoutConstraints(Math.max(0.01, jsonBounds.width), Math.max(0.01, jsonBounds.height));
    } catch (_e) {
      /* text / vector-like nodes may reject direct resize */
    }
  }

  const wantDoc = jsonBannerPointToDocument(bannerRootFrame, jsonBounds.x, jsonBounds.y);
  const beforeAbs = getAbsXY(node);
  const dDocX = wantDoc.x - beforeAbs.x;
  const dDocY = wantDoc.y - beforeAbs.y;
  const label = String(jsonLabel || node.name || node.id || "?");

  if (Math.abs(dDocX) < 1e-4 && Math.abs(dDocY) < 1e-4) {
    console.log("[bounds-fix]", label, {
      target: { x: jsonBounds.x, y: jsonBounds.y, width: jsonBounds.width, height: jsonBounds.height },
      before: beforeAbs,
      after: getAbsXY(node),
      size: "applied",
    });
    return;
  }

  const p = node.parent;
  if (p && typeof p.absoluteTransform === "object" && p.absoluteTransform) {
    const M = p.absoluteTransform;
    const a = M[0][0];
    const b = M[1][0];
    const c = M[0][1];
    const d = M[1][1];
    const det = a * d - b * c;
    if (Math.abs(det) > 1e-9) {
      const lx = (d * dDocX - c * dDocY) / det;
      const ly = (-b * dDocX + a * dDocY) / det;
      node.x += lx;
      node.y += ly;
    } else {
      node.x += dDocX;
      node.y += dDocY;
    }
  } else {
    node.x += dDocX;
    node.y += dDocY;
  }

  console.log("[bounds-fix]", label, {
    target: { x: jsonBounds.x, y: jsonBounds.y, width: jsonBounds.width, height: jsonBounds.height },
    before: beforeAbs,
    after: getAbsXY(node),
  });
}

/**
 * Preorder: align each Figma node to ``jsonNode.bounds`` in banner space, then recurse JSON children.
 * Matches Figma children by ``originalNodeId`` under ``figmaNode``, then ``byId`` fallback.
 */
function applyFinalBoundsRecursive(figmaNode, jsonNode, bannerRootFrame, byId) {
  let corrected = 0;
  let skippedChildren = 0;

  if (figmaNode && jsonNode && jsonNode.bounds && typeof jsonNode.bounds === "object") {
    const jName = String(jsonNode.name || jsonNode.id || figmaNode.name || "").trim();
    forceAbsolutePosition(figmaNode, bannerRootFrame, jsonNode.bounds, jName);
    corrected++;
  }

  const jch = Array.isArray(jsonNode.children) ? jsonNode.children : [];
  for (let i = 0; i < jch.length; i++) {
    const jc = jch[i];
    if (!jc || typeof jc !== "object") continue;
    const cid = String(jc.id || "").trim();
    if (!cid) continue;

    let fc = null;
    if (figmaNode && "children" in figmaNode && Array.isArray(figmaNode.children)) {
      for (let k = 0; k < figmaNode.children.length; k++) {
        const cand = figmaNode.children[k];
        let oid = "";
        try {
          oid = String(cand.getPluginData("originalNodeId") || "").trim();
        } catch (_e) {
          oid = "";
        }
        if (oid === cid) {
          fc = cand;
          break;
        }
      }
    }
    if (!fc && byId.has(cid)) {
      const cand = byId.get(cid);
      if (cand && figmaNode && cand.parent && cand.parent.id === figmaNode.id) {
        fc = cand;
      }
    }
    if (!fc) {
      skippedChildren++;
      continue;
    }
    const sub = applyFinalBoundsRecursive(fc, jc, bannerRootFrame, byId);
    corrected += sub.corrected;
    skippedChildren += sub.skippedChildren;
  }

  return { corrected, skippedChildren };
}

/**
 * Final pass: force every mapped node so document geometry matches ``final_json`` banner-space bounds.
 */
function applyFinalAbsoluteBoundsCorrection(jsonTree, figmaRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(figmaRoot);
  return applyFinalBoundsRecursive(figmaRoot, jsonTree, figmaRoot, byId);
}

/**
 * Full clone reconstruction: hierarchy → prune extras → child order → stamp ids → drop strays →
 * re-order → layout from JSON bounds → empty cleanup → **absolute bounds correction** (document nudge).
 */
function applyFinalJsonCloneReconstruction(jsonTree, cloneRoot) {
  const preStampReport = stampCloneOriginalIdsFromJson(jsonTree, cloneRoot);
  const detachReport = detachInstancesForFinalJson(jsonTree, cloneRoot);
  const hierarchyReport = syncCloneHierarchyToJsonTree(jsonTree, cloneRoot);
  const pruneReport = pruneClonedNodesMissingFromFinalJson(jsonTree, cloneRoot);
  const reorderAfterPrune = reorderCloneChildrenPerFinalJson(jsonTree, cloneRoot);
  const stampReport = stampCloneOriginalIdsFromJson(jsonTree, cloneRoot);
  const strayReport = removeStrayFigmaChildrenNotInJson(jsonTree, cloneRoot);
  const reorderAfterStray = reorderCloneChildrenPerFinalJson(jsonTree, cloneRoot);
  const layoutReport = applyFinalJsonAbsoluteLayout(jsonTree, cloneRoot);
  const emptyReport = removeEmptyFramesUnder(cloneRoot);
  const boundsFixReport = applyFinalAbsoluteBoundsCorrection(jsonTree, cloneRoot);
  const pathNameReport = applyJsonTreeNamesByPath(jsonTree, cloneRoot);
  const zeroSizeReport = removeEmptyZeroSizeNodes(cloneRoot);
  return {
    preStampReport,
    detachReport,
    hierarchyReport,
    pruneReport,
    reorderAfterPrune,
    stampReport,
    strayReport,
    reorderAfterStray,
    layoutReport,
    emptyReport,
    boundsFixReport,
    pathNameReport,
    zeroSizeReport,
  };
}

/**
 * Apply ``bounds`` from ``jsonTree`` as **absolute** coordinates in banner-root space, converting to
 * parent-local coordinates by walking the JSON tree:
 *
 *   node.x = child.bounds.x - parent.bounds.x
 *   node.y = child.bounds.y - parent.bounds.y
 *
 * Direct children of ``banner_root`` use that same formula with ``parent`` = root (typically
 * origin 0,0). No special cases for BOOLEAN_OPERATION / GROUP / FRAME beyond Figma API limits.
 *
 * Resize runs **before** x/y so boolean/group resize does not reset position to parent origin.
 */
function applyFinalJsonAbsoluteLayout(jsonTree, cloneRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
  let applied = 0;
  let skipped = 0;

  function walk(item, parentJsonNode) {
    if (!item || typeof item !== "object") return;
    const abs = jsonBounds(item);
    const id = String(item.id || "").trim();
    const node = id ? byId.get(id) : null;

    if (node && cloneRoot && node.id === cloneRoot.id) {
      applyJsonClippingBehavior(node, item);
      if (typeof abs.width === "number" && typeof abs.height === "number") {
        resizeNodeIfPossible(node, abs.width, abs.height);
        applied++;
      }
      applyJsonClippingBehavior(node, item);
    } else if (node) {
      applyBoundsFromAbsoluteJson(node, item, parentJsonNode);
      applied++;
    } else if (id) {
      skipped++;
    }

    const passToChildren = id ? item : parentJsonNode;
    const kids = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < kids.length; i++) {
      walk(kids[i], passToChildren);
    }
  }

  walk(jsonTree, null);

  const indexIds = collectFinalJsonNodeIds(jsonTree).size;
  return { applied, skipped, indexIds, mapSize: byId.size };
}

/**
 * Remove clone subtrees whose ``originalNodeId`` is absent from ``final_json`` / ``semantic_json``.
 * Lifts remaining children to the parent before removal so valid nodes are not deleted with a wrapper.
 */
function pruneClonedNodesMissingFromFinalJson(jsonTree, cloneRoot) {
  const allowed = collectFinalJsonNodeIds(jsonTree);
  let removed = 0;

  function visitPost(node) {
    if (!node || node === cloneRoot || !("children" in node) || !Array.isArray(node.children)) return;
    const kids = [...node.children];
    for (let i = 0; i < kids.length; i++) {
      visitPost(kids[i]);
    }
    const kids2 = [...node.children];
    for (let j = 0; j < kids2.length; j++) {
      const c = kids2[j];
      let oid = "";
      try {
        oid = String(c.getPluginData("originalNodeId") || "").trim();
      } catch (_e) {
        oid = "";
      }
      if (!oid || allowed.has(oid)) continue;
      try {
        c.remove();
        removed++;
      } catch (e2) {
        console.warn("prune: remove failed", c && c.id, e2);
      }
    }
  }

  visitPost(cloneRoot);
  return { removed };
}

/**
 * Remove empty wrappers and zero-size empty nodes under ``root`` (e.g. stale logo wrappers after reparent + prune).
 */
function removeEmptyFramesUnder(root) {
  let total = 0;
  let rounds = 0;
  let changed = true;
  while (changed && rounds < 64) {
    changed = false;
    rounds++;
    const toRemove = [];

    function collectEmpty(node) {
      if (!node) return;
      const hasChildrenArray = "children" in node && Array.isArray(node.children);
      const childCount = hasChildrenArray ? node.children.length : 0;
      if (hasChildrenArray) {
        for (let i = 0; i < node.children.length; i++) {
          collectEmpty(node.children[i]);
        }
      }
      const t = node.type;
      const isEmptyWrapper = (t === "FRAME" || t === "GROUP" || t === "INSTANCE") && childCount === 0;
      const isZeroSizeEmpty =
        childCount === 0 &&
        "width" in node &&
        "height" in node &&
        Math.abs(Number(node.width) || 0) <= 0.01 &&
        Math.abs(Number(node.height) || 0) <= 0.01;
      if (node !== root && (isEmptyWrapper || isZeroSizeEmpty)) {
        toRemove.push(node);
      }
    }

    collectEmpty(root);
    for (let r = 0; r < toRemove.length; r++) {
      try {
        toRemove[r].remove();
        total++;
        changed = true;
      } catch (e) {
        console.warn("removeEmptyFramesUnder: remove failed", toRemove[r] && toRemove[r].id, e);
      }
    }
  }
  return { removed: total, rounds };
}

/**
 * Remove any empty zero-size node under ``root``. This catches stale boolean/vector logo shells that
 * survive hierarchy reconstruction but are absent from ``final_json``.
 */
function removeEmptyZeroSizeNodes(root) {
  let removed = 0;
  let rounds = 0;
  let changed = true;

  while (changed && rounds < 64) {
    changed = false;
    rounds++;
    const toRemove = [];

    function collect(node) {
      if (!node) return;
      const hasChildrenArray = "children" in node && Array.isArray(node.children);
      if (hasChildrenArray) {
        for (let i = 0; i < node.children.length; i++) {
          collect(node.children[i]);
        }
      }
      const childCount = hasChildrenArray ? node.children.length : 0;
      const hasSize = "width" in node && "height" in node;
      const width = hasSize ? Number(node.width) || 0 : 0;
      const height = hasSize ? Number(node.height) || 0 : 0;
      if (node !== root && hasSize && childCount === 0 && width <= 0.01 && height <= 0.01) {
        toRemove.push(node);
      }
    }

    collect(root);
    for (let i = 0; i < toRemove.length; i++) {
      try {
        toRemove[i].remove();
        removed++;
        changed = true;
      } catch (e) {
        console.warn("removeEmptyZeroSizeNodes: remove failed", toRemove[i] && toRemove[i].id, e);
      }
    }
  }

  return { removed, rounds };
}

/**
 * All id-like keys on a JSON tree node that may match ``originalNodeId`` / Figma serialization.
 */
function jsonSourceIdKeys(item) {
  if (!item || typeof item !== "object") return [];
  const seen = new Set();
  const out = [];
  for (const k of ["id", "source_figma_id", "sourceFigmaId", "figma_node_id", "node_id"]) {
    const v = item[k];
    if (v == null) continue;
    const s = String(v).trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

/**
 * Find the Figma node for a ``final_json`` row. ``byId`` maps **source** ids (``originalNodeId`` / JSON id),
 * never the post-clone Figma ``node.id``, to the cloned node. Falls back to ``path`` under ``cloneRoot``.
 */
function resolveFigmaNodeForJsonItem(item, byId, cloneRoot) {
  if (!item || typeof item !== "object" || !cloneRoot) return null;
  for (const key of jsonSourceIdKeys(item)) {
    if (byId.has(key)) return byId.get(key);
  }
  const p = item.path;
  if (p == null) return null;
  const ps = String(p).trim();
  if (!ps) return cloneRoot;
  return getNodeByPath(cloneRoot, ps);
}

/**
 * Apply semantic layer name from backend (``name`` / ``semantic_name`` / ``role``).
 * Sets ``node.name`` directly so INSTANCE / all types receive the label when the API allows.
 */
function applySemanticNameToFigmaNode(node, rawName) {
  if (!node || rawName == null) return false;
  const clean = sanitizeLayerName(String(rawName).trim());
  if (!clean) return false;
  try {
    node.name = clean;
    node.setPluginData("semanticName", clean);
    return true;
  } catch (e) {
    console.warn("applySemanticNameToFigmaNode failed", node && node.id, e);
    return false;
  }
}

/**
 * Walk ``final_json`` and rename nodes in ``cloneRoot`` using ``originalNodeId`` plugin data
 * (``collectClonedNodesByOriginalId``) — keys are **source** ids, not new Figma ids.
 */
function applyJsonTreeNamesByOriginalIds(jsonTree, cloneRoot) {
  const { map: byId, mapped } = collectClonedNodesByOriginalId(cloneRoot);
  let renamed = 0;
  const missing = [];

  function walk(item) {
    if (!item || typeof item !== "object") return;
    const nm =
      String(item.name || "").trim() ||
      String(getSemanticName(item) || "").trim();
    const keys = jsonSourceIdKeys(item);
    const primaryId = keys.length ? keys[0] : "";

    if (nm) {
      const node = resolveFigmaNodeForJsonItem(item, byId, cloneRoot);
      if (node) {
        if (applySemanticNameToFigmaNode(node, nm)) renamed++;
      } else if (primaryId) {
        missing.push(primaryId);
      }
    }
    const kids = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < kids.length; i++) walk(kids[i]);
  }

  walk(jsonTree);
  return { renamed, missing, mapped };
}

/**
 * Path-only fallback for semantic names. This does not depend on cloned node ids or ``originalNodeId``:
 * it indexes backend JSON by ``path`` (or derived child-index path) and walks the Figma clone by current path.
 */
function applyJsonTreeNamesByPath(jsonTree, cloneRoot) {
  const byPath = new Map();
  let renamed = 0;
  let stamped = 0;
  const missing = [];

  function index(jNode, derivedPath) {
    if (!jNode || typeof jNode !== "object") return;
    const explicitPath = jNode.path !== undefined && jNode.path !== null ? String(jNode.path).trim() : "";
    byPath.set(explicitPath || derivedPath, jNode);
    const kids = Array.isArray(jNode.children) ? jNode.children : [];
    for (let i = 0; i < kids.length; i++) {
      index(kids[i], derivedPath ? `${derivedPath}/${i}` : String(i));
    }
  }

  function stampSourceId(figmaNode, jsonNode) {
    const jid = String(jsonNode && jsonNode.id || "").trim();
    if (!jid || !figmaNode) return;
    try {
      const existing = String(figmaNode.getPluginData("originalNodeId") || "").trim();
      if (!existing) {
        figmaNode.setPluginData("originalNodeId", jid);
        stamped++;
      } else if (existing !== jid) {
        console.warn("applyJsonTreeNamesByPath: preserve existing originalNodeId", {
          figmaId: figmaNode.id,
          existing,
          wanted: jid,
          jsonName: jsonNode && jsonNode.name,
          figmaName: figmaNode.name,
        });
      }
    } catch (e) {
      console.warn("applyJsonTreeNamesByPath: set originalNodeId failed", jid, e);
    }
  }

  function walk(figmaNode, path) {
    if (!figmaNode) return;
    const jsonNode = byPath.get(path);
    if (jsonNode) {
      const nm =
        String(jsonNode.name || "").trim() ||
        String(getSemanticName(jsonNode) || "").trim();
      if (nm && applySemanticNameToFigmaNode(figmaNode, nm)) renamed++;
      stampSourceId(figmaNode, jsonNode);
    } else {
      missing.push(path);
    }

    if ("children" in figmaNode && Array.isArray(figmaNode.children)) {
      for (let i = 0; i < figmaNode.children.length; i++) {
        walk(figmaNode.children[i], path ? `${path}/${i}` : String(i));
      }
    }
  }

  index(jsonTree, "");
  walk(cloneRoot, "");
  const zeroSizeReport = removeEmptyZeroSizeNodes(cloneRoot);
  return { renamed, stamped, missing, zeroSizeRemoved: zeroSizeReport.removed };
}

function applyExactNodeNamesFromJson(jsonTree, cloneRoot) {
  const { map: byId } = collectClonedNodesByOriginalId(cloneRoot);
  const pathNodeMap = buildPathNodeMap(cloneRoot);
  let renamed = 0;
  const missing = [];

  function resolve(item, path) {
    if (!item || typeof item !== "object") return null;
    const id = String(item.id || "").trim();
    if (id && byId.has(id)) return byId.get(id);
    if (path && pathNodeMap.has(path)) return pathNodeMap.get(path);
    if (!path) return cloneRoot;
    return null;
  }

  function walk(item, path) {
    if (!item || typeof item !== "object") return;
    const node = resolve(item, path);
    const wantedName = String(item.name || "").trim();
    if (node && wantedName) {
      if (applySemanticNameToFigmaNode(node, wantedName)) renamed++;
    } else if (!node && (item.id || path)) {
      missing.push(String(item.id || path));
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < children.length; i++) {
      const childPath = path ? `${path}/${i}` : String(i);
      walk(children[i], childPath);
    }
  }

  walk(jsonTree, "");
  return { renamed, missing };
}

/**
 * Index backend ``final_json`` / ``semantic_json`` by stable **source** ids and ``path`` (never clone ids).
 * Values are the JSON objects so callers can read ``.name``.
 */
function indexFinalJsonSemantics(jsonTree) {
  const semanticBySourceId = new Map();
  const semanticByPath = new Map();

  function indexSemantic(node) {
    if (!node || typeof node !== "object") return;
    const id = node.id != null ? String(node.id).trim() : "";
    if (id) semanticBySourceId.set(id, node);
    const sf = node.source_figma_id != null ? String(node.source_figma_id).trim() : "";
    if (sf) semanticBySourceId.set(sf, node);
    const sf2 = node.sourceFigmaId != null ? String(node.sourceFigmaId).trim() : "";
    if (sf2) semanticBySourceId.set(sf2, node);
    if (node.path !== undefined && node.path !== null) {
      const pk = String(node.path).trim();
      semanticByPath.set(pk, node);
    }
    const kids = Array.isArray(node.children) ? node.children : [];
    for (let i = 0; i < kids.length; i++) indexSemantic(kids[i]);
  }

  indexSemantic(jsonTree);
  return { semanticBySourceId, semanticByPath };
}

/**
 * Pair ``final_json`` nodes with Figma nodes by **parallel tree walk** (child index order).
 * Does not look up semantics using ``figmaNode.id`` — only ``jsonNode`` shape + ``jsonNode.name``.
 */
function applySemanticNamesParallelTreeWalk(jsonTree, figmaRoot) {
  let renamed = 0;

  function walk(jsonNode, figmaNode) {
    if (!jsonNode || typeof jsonNode !== "object" || !figmaNode) return;
    const nm =
      String(jsonNode.name || "").trim() ||
      String(getSemanticName(jsonNode) || "").trim();
    if (nm && applySemanticNameToFigmaNode(figmaNode, nm)) renamed++;

    const jch = Array.isArray(jsonNode.children) ? jsonNode.children : [];
    if (!("children" in figmaNode) || !Array.isArray(figmaNode.children)) return;
    const fch = figmaNode.children;
    const n = Math.min(jch.length, fch.length);
    for (let i = 0; i < n; i++) walk(jch[i], fch[i]);
  }

  walk(jsonTree, figmaRoot);
  return { renamed };
}

/**
 * Apply semantic names using **only** ``originalNodeId`` (stamped source / JSON id), never ``node.id``.
 */
function applySemanticNamesFromStampedSourceIds(maps, figmaRoot) {
  const { semanticBySourceId, semanticByPath } = maps;
  let renamed = 0;

  function visit(figmaNode) {
    let oid = "";
    try {
      oid = String(figmaNode.getPluginData("originalNodeId") || "").trim();
    } catch (_e) {
      oid = "";
    }
    let semantic = null;
    if (oid) {
      semantic = semanticBySourceId.get(oid) || null;
      if (!semantic) semantic = semanticByPath.get(oid) || null;
    }
    if (semantic) {
      const nm =
        String(semantic.name || "").trim() ||
        String(getSemanticName(semantic) || "").trim();
      if (nm && applySemanticNameToFigmaNode(figmaNode, nm)) renamed++;
    }
    if ("children" in figmaNode && Array.isArray(figmaNode.children)) {
      for (let i = 0; i < figmaNode.children.length; i++) visit(figmaNode.children[i]);
    }
  }

  visit(figmaRoot);
  return { renamed };
}

/**
 * Warn when layer names still look like raw Figma defaults but indexed semantics expect a label.
 */
function warnRawSemanticLayerNames(figmaRoot, maps) {
  const { semanticBySourceId, semanticByPath } = maps;
  const bad = [];

  function visit(node) {
    if (!node) return;
    const nm = String(node.name || "");
    const rawNumeric = /^[0-9]+$/.test(nm);
    const rawGroup = /^Group\s+/i.test(nm);
    if (!rawNumeric && !rawGroup) {
      if ("children" in node && Array.isArray(node.children)) {
        for (let i = 0; i < node.children.length; i++) visit(node.children[i]);
      }
      return;
    }
    let oid = "";
    try {
      oid = String(node.getPluginData("originalNodeId") || "").trim();
    } catch (_e) {
      oid = "";
    }
    const semantic = oid
      ? semanticBySourceId.get(oid) || semanticByPath.get(oid)
      : null;
    const expected = semantic ? String(semantic.name || "").trim() : "";
    if (expected && expected !== nm) {
      bad.push({ figmaId: node.id, originalNodeId: oid, name: nm, expected });
    } else if (!expected && rawNumeric) {
      bad.push({ figmaId: node.id, originalNodeId: oid || "(none)", name: nm, expected: "(no semantic row)" });
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (let j = 0; j < node.children.length; j++) visit(node.children[j]);
    }
  }

  visit(figmaRoot);
  if (bad.length) {
    console.warn("[semantic-name-fail]", bad.map((b) => b.name), bad);
  }
  return { badCount: bad.length, samples: bad.slice(0, 12) };
}

/**
 * Last-chance semantic labels: detach INSTANCEs, re-stamp source ids, then apply names using
 * **indexed** ``final_json`` (source id / path) + parallel tree walk — never semantic lookup by new Figma id.
 */
function finalizeSemanticLayerNamesFromJson(jsonTree, cloneRoot) {
  const detachReport = detachInstancesForFinalJson(jsonTree, cloneRoot);
  stampCloneOriginalIdsFromJson(jsonTree, cloneRoot);

  const maps = indexFinalJsonSemantics(jsonTree);
  let renamed = 0;
  renamed += applySemanticNamesParallelTreeWalk(jsonTree, cloneRoot).renamed;
  renamed += applySemanticNamesFromStampedSourceIds(maps, cloneRoot).renamed;

  const nameReport = applyJsonTreeNamesByOriginalIds(jsonTree, cloneRoot);
  renamed += nameReport.renamed;
  const pathNameReport = applyJsonTreeNamesByPath(jsonTree, cloneRoot);
  renamed += pathNameReport.renamed;
  const zeroSizeReport = removeEmptyZeroSizeNodes(cloneRoot);

  const validation = warnRawSemanticLayerNames(cloneRoot, maps);

  return {
    renamed,
    missing: nameReport.missing,
    mapped: nameReport.mapped,
    detached: detachReport.detached,
    pathRenamed: pathNameReport.renamed,
    pathStamped: pathNameReport.stamped,
    zeroSizeRemoved: zeroSizeReport.removed,
    validationBadCount: validation.badCount,
  };
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
    if (data && typeof data === "object" && data.supported === false && Number(data.category) === -1) {
      return data;
    }
    throw new Error("Pipeline backend returned invalid JSON or missing final_json.");
  }
  return data;
}

async function callLayoutTransformer(backendUrl, rawJson, targetResolution, endpointPath) {
  const url = String(backendUrl || "").trim().replace(/\/+$/, "");
  if (!url) throw new Error("Backend URL is empty.");
  const endpoint = endpointPath || "/api/layout-transformer";
  const response = await fetch(url + endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_json: rawJson,
      target_width: targetResolution.width,
      target_height: targetResolution.height,
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
    throw new Error(`Layout transformer backend failed: ${detail}`);
  }
  if (!data || typeof data !== "object" || !data.final_json) {
    throw new Error("Layout transformer returned invalid JSON or missing final_json.");
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

async function loadJsonFontName(fontName) {
  if (fontName && typeof fontName === "object" && fontName.family && fontName.style) {
    const wanted = { family: String(fontName.family), style: String(fontName.style) };
    try {
      await figma.loadFontAsync(wanted);
      return wanted;
    } catch (e) {
      console.warn("loadJsonFontName: JSON font unavailable; using Inter for node creation", wanted, e);
    }
  }
  const fallback = { family: "Inter", style: "Regular" };
  try {
    await figma.loadFontAsync(fallback);
    return fallback;
  } catch (fallbackError) {
    console.warn("loadJsonFontName: Inter fallback unavailable; using default text font", fallbackError);
    return null;
  }
}

/** Generate pipeline: headline always uses the boldest available style for the JSON font family. */
const HEADLINE_BOLD_STYLE_CANDIDATES = ["Black", "Bold", "Heavy", "Extra Bold", "Demi Bold", "Semi Bold"];

async function loadHeadlineBoldFontName(fontName) {
  const family =
    fontName && typeof fontName === "object" && fontName.family
      ? String(fontName.family)
      : "YS Geo";
  for (const style of HEADLINE_BOLD_STYLE_CANDIDATES) {
    const candidate = { family, style };
    try {
      await figma.loadFontAsync(candidate);
      return candidate;
    } catch (_e) {
      /* try next weight */
    }
  }
  return loadJsonFontName(fontName || { family, style: "Bold" });
}

function applyJsonPropertyIfPresent(node, item, key) {
  if (!node || !item || !(key in item) || !(key in node)) return false;
  const value = jsonSafeValue(item[key]);
  if (value === undefined) return false;
  try {
    node[key] = value;
    return true;
  } catch (e) {
    console.warn(`applyJsonPropertyIfPresent: failed applying ${key}`, node && node.name, e);
    return false;
  }
}

function applyJsonVisualStyle(node, item) {
  if (!node || !item) return 0;
  let applied = 0;
  for (const key of ["fills", "strokes", "strokeWeight", "effects", "blendMode", "cornerRadius"]) {
    if (applyJsonPropertyIfPresent(node, item, key)) applied++;
  }
  return applied;
}

function applyJsonTextStyle(node, item) {
  if (!node || !item) return 0;
  let applied = 0;
  for (const key of [
    "lineHeight",
    "letterSpacing",
    "paragraphSpacing",
    "paragraphIndent",
    "textCase",
    "textDecoration",
    "fills",
    "opacity",
  ]) {
    if (applyJsonPropertyIfPresent(node, item, key)) applied++;
  }
  return applied;
}

function applyUniformTextRangeValue(node, rangeSetterName, propertyName, value, role) {
  if (!node || !("characters" in node)) return false;
  const length = String(node.characters || "").length;
  if (length <= 0) return false;
  try {
    if (typeof node[rangeSetterName] === "function") {
      node[rangeSetterName](0, length, jsonSafeValue(value));
      return true;
    }
  } catch (e) {
    console.warn(`[TEXT_RANGE_${propertyName}_APPLY_FAILED]`, role, e);
  }
  return false;
}

async function loadAllFontsForTextNode(node, role) {
  if (!node || !("characters" in node)) return;
  const fonts = [];
  try {
    if (typeof node.getRangeAllFontNames === "function") {
      const rangeFonts = node.getRangeAllFontNames(0, node.characters.length);
      if (Array.isArray(rangeFonts)) fonts.push(...rangeFonts);
    }
  } catch (e) {
    console.warn("[TEXT_RANGE_FONT_SCAN_FAILED]", role, e);
  }
  const fontName = node.fontName;
  if (fontName && typeof fontName === "object" && fontName.family && fontName.style) {
    fonts.push(fontName);
  }
  const seen = new Set();
  for (const font of fonts) {
    if (!font || typeof font !== "object" || !font.family || !font.style) continue;
    const key = `${font.family}\n${font.style}`;
    if (seen.has(key)) continue;
    seen.add(key);
    try {
      await figma.loadFontAsync({ family: String(font.family), style: String(font.style) });
    } catch (e) {
      console.warn("[TEXT_EXISTING_FONT_LOAD_FAILED]", role, font, e);
    }
  }
}

async function applyFinalJsonTextStyle(node, item, parentJsonNode, sourceContentMap) {
  if (!node || !item || !("characters" in node)) return 0;
  const role = semanticRoleName(item);
  const parentRole = semanticRoleName(parentJsonNode);
  const isAgeBadgeText = isAgeBadgeTextRole(role, parentRole);
  const sourceChars = resolveSourceCharactersForJsonText(item, sourceContentMap);
  const requestedAutoResize = typeof item.textAutoResize === "string" ? item.textAutoResize : null;
  let loadedJsonFontName = null;
  await loadAllFontsForTextNode(node, role);
  if (item.fontName && typeof item.fontName === "object" && item.fontName.family && item.fontName.style) {
    const fontName =
      role === "headline"
        ? await loadHeadlineBoldFontName(item.fontName)
        : { family: String(item.fontName.family), style: String(item.fontName.style) };
    if (fontName) {
      try {
        await figma.loadFontAsync(fontName);
        node.fontName = fontName;
        loadedJsonFontName = fontName;
      } catch (e) {
        console.warn("[TEXT_JSON_FONT_LOAD_FAILED]", role, fontName, e);
      }
    }
  } else if (role === "headline") {
    const fontName = await loadHeadlineBoldFontName(null);
    if (fontName) {
      try {
        await figma.loadFontAsync(fontName);
        node.fontName = fontName;
        loadedJsonFontName = fontName;
      } catch (e) {
        console.warn("[TEXT_HEADLINE_BOLD_FONT_FAILED]", role, fontName, e);
      }
    }
  }

  const rawCharacters = String(sourceChars != null ? sourceChars : item.characters || "");
  const finalCharacters = isAgeBadgeText ? singleLineText(rawCharacters) : rawCharacters;
  try {
    node.characters = finalCharacters;
  } catch (e) {
    console.warn("[TEXT_CHARACTERS_APPLY_FAILED]", role, e);
  }
  if ("textAutoResize" in node) {
    try {
      node.textAutoResize = "NONE";
    } catch (_e) {
      /* fixed sizing is best-effort before bounds placement */
    }
  }
  if (loadedJsonFontName) {
    applyUniformTextRangeValue(node, "setRangeFontName", "FONT_NAME", loadedJsonFontName, role);
  }

  const b = item.bounds && typeof item.bounds === "object" ? item.bounds : null;
  const pb = parentJsonNode && parentJsonNode.bounds && typeof parentJsonNode.bounds === "object" ? parentJsonNode.bounds : null;
  if (b && typeof b.width === "number" && typeof b.height === "number") {
    resizeNodeIfPossible(node, b.width, b.height);
  }
  if (b) {
    if (typeof b.x === "number") node.x = b.x - (pb && typeof pb.x === "number" ? pb.x : 0);
    if (typeof b.y === "number") node.y = b.y - (pb && typeof pb.y === "number" ? pb.y : 0);
  }

  if (typeof item.fontSize === "number" && Number.isFinite(item.fontSize)) {
    try {
      node.fontSize = Math.max(1, item.fontSize);
    } catch (e) {
      console.warn("[TEXT_FONT_SIZE_APPLY_FAILED]", role, item.fontSize, e);
    }
    applyUniformTextRangeValue(node, "setRangeFontSize", "FONT_SIZE", Math.max(1, item.fontSize), role);
  }
  const targetTextAlignHorizontal = textAlignHorizontalForTarget(item, sourceContentMap);
  if (targetTextAlignHorizontal) {
    try {
      node.textAlignHorizontal = targetTextAlignHorizontal;
    } catch (e) {
      console.warn("[TEXT_ALIGN_H_APPLY_FAILED]", role, targetTextAlignHorizontal, e);
    }
  }
  const targetTextAlignVertical = textAlignVerticalForTarget(item, sourceContentMap);
  if (targetTextAlignVertical) {
    try {
      node.textAlignVertical = targetTextAlignVertical;
    } catch (e) {
      console.warn("[TEXT_ALIGN_V_APPLY_FAILED]", role, targetTextAlignVertical, e);
    }
  }
  if (applyJsonPropertyIfPresent(node, item, "lineHeight")) {
    applyUniformTextRangeValue(node, "setRangeLineHeight", "LINE_HEIGHT", item.lineHeight, role);
  }
  if (applyJsonPropertyIfPresent(node, item, "letterSpacing")) {
    applyUniformTextRangeValue(node, "setRangeLetterSpacing", "LETTER_SPACING", item.letterSpacing, role);
  }
  if (applyJsonPropertyIfPresent(node, item, "fills")) {
    applyUniformTextRangeValue(node, "setRangeFills", "FILLS", item.fills, role);
  }
  applyJsonPropertyIfPresent(node, item, "opacity");
  applyJsonPropertyIfPresent(node, item, "textCase");
  applyJsonPropertyIfPresent(node, item, "textDecoration");
  applyJsonPropertyIfPresent(node, item, "paragraphSpacing");
  applyJsonPropertyIfPresent(node, item, "paragraphIndent");
  if ("textAutoResize" in node && requestedAutoResize && !isAgeBadgeText) {
    try {
      node.textAutoResize = requestedAutoResize;
    } catch (_e) {
      /* textAutoResize is best-effort across text node variants */
    }
  }
  ensureAgeBadgeSingleLine(node, isAgeBadgeText);

  if (role === "headline") {
    const boldFont = await loadHeadlineBoldFontName(loadedJsonFontName || item.fontName);
    if (boldFont) {
      try {
        await figma.loadFontAsync(boldFont);
        node.fontName = boldFont;
        applyUniformTextRangeValue(node, "setRangeFontName", "FONT_NAME", boldFont, role);
        applyExactTextFontSize(node, item.fontSize, role);
      } catch (e) {
        console.warn("[TEXT_HEADLINE_BOLD_FONT_FAILED]", role, boldFont, e);
      }
    }
  }

  console.log("[TEXT_CONTENT_POLICY]", item.name, {
    finalJsonCharacters: item.characters,
    sourceCharacters: sourceChars,
    usedCharacters: node.characters,
    x: node.x,
    y: node.y,
    w: node.width,
    h: node.height,
    fontSize: node.fontSize,
    fontName: node.fontName,
    alignH: node.textAlignHorizontal,
    alignV: node.textAlignVertical,
    autoResize: node.textAutoResize,
  });
  return 1;
}

function shouldDisableClippingForJsonItem(item) {
  const name = String(item && item.name ? item.name : "").trim();
  return name === "brand_group" || name === "headline_group" || name === "offer_group";
}

function isJsonRootFrame(item) {
  if (!item || typeof item !== "object") return false;
  const name = String(item.name || "").trim();
  const path = item.path != null ? String(item.path).trim() : "";
  return path === "" || name.indexOf("banner_root") !== -1;
}

function applyJsonClippingBehavior(figmaNode, jsonNode) {
  if (!figmaNode || !jsonNode || !("clipsContent" in figmaNode)) return;
  try {
    if (isJsonRootFrame(jsonNode)) {
      figmaNode.clipsContent = true;
    } else if (typeof jsonNode.clipsContent === "boolean") {
      figmaNode.clipsContent = jsonNode.clipsContent;
    } else if (shouldDisableClippingForJsonItem(jsonNode)) {
      figmaNode.clipsContent = false;
    }
  } catch (_e) {
    /* some node types do not support clipsContent */
  }
}

function applyBoundsFromAbsoluteJson(figmaNode, jsonNode, parentJsonNode) {
  if (!figmaNode || !jsonNode || typeof jsonNode !== "object") return;
  applyJsonClippingBehavior(figmaNode, jsonNode);
  const b = jsonBounds(jsonNode);
  const pb = parentJsonNode && parentJsonNode.bounds && typeof parentJsonNode.bounds === "object"
    ? jsonBounds(parentJsonNode)
    : null;

  figmaNode.resizeWithoutConstraints(
    Math.max(0.01, b.width),
    Math.max(0.01, b.height)
  );

  if (pb) {
    figmaNode.x = b.x - pb.x;
    figmaNode.y = b.y - pb.y;
  } else {
    figmaNode.x = b.x;
    figmaNode.y = b.y;
  }
  applyJsonClippingBehavior(figmaNode, jsonNode);
}

function figmaNodeTypeFromJson(item, isRoot) {
  const type = normalizeType(item && item.type ? item.type : "");
  const hasChildren = Array.isArray(item && item.children) && item.children.length > 0;
  if (isRoot || hasChildren || type === "frame" || type === "group") return "FRAME";
  if (type === "text") return "TEXT";
  if (type === "star") return "STAR";
  return "RECTANGLE";
}

async function createNodeFromJsonItem(item, parent, parentBounds, isRoot, sourceContentMap) {
  if (!item || typeof item !== "object") return null;
  const bounds = jsonBounds(item);
  const clonedVisual = shouldCloneSourceVisualForJsonItem(item, isRoot)
    ? cloneSourceVisualIntoJson(parent, item, parentBounds, sourceContentMap)
    : null;
  if (clonedVisual) return clonedVisual;

  const figmaType = figmaNodeTypeFromJson(item, isRoot);
  let node;

  if (figmaType === "TEXT") {
    const fontName = await loadJsonFontName(item.fontName);
    node = figma.createText();
    if (fontName) node.fontName = fontName;
    node.characters = String(item.characters || item.name || "Text");
    if ("textAutoResize" in node) node.textAutoResize = "NONE";
  } else if (figmaType === "FRAME") {
    node = figma.createFrame();
    node.layoutMode = "NONE";
    node.clipsContent = false;
    node.fills = isRoot ? [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }] : [];
  } else if (figmaType === "STAR" && typeof figma.createStar === "function") {
    node = figma.createStar();
    if (applyJsonVisualStyle(node, item) === 0) {
      node.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
    }
  } else {
    node = figma.createRectangle();
    if (applyJsonVisualStyle(node, item) === 0) {
      node.fills = [{ type: "SOLID", color: { r: 0.72, g: 0.78, b: 0.86 }, opacity: 0.28 }];
      node.strokes = [{ type: "SOLID", color: { r: 0.34, g: 0.45, b: 0.58 } }];
      node.strokeWeight = 1;
    }
  }

  node.name = String(item.name || item.type || "json_node");
  if (typeof item.visible === "boolean") {
    try {
      node.visible = item.visible;
    } catch (_e) {
      /* visibility is best-effort across node variants */
    }
  }
  if (typeof item.opacity === "number" && Number.isFinite(item.opacity)) {
    try {
      node.opacity = item.opacity;
    } catch (_e) {
      /* opacity is best-effort across node variants */
    }
  }
  try {
    if (item.id) node.setPluginData("originalNodeId", String(item.id));
    node.setPluginData("semanticName", String(item.name || ""));
  } catch (_e) {
    /* plugin data best-effort */
  }
  applyJsonClippingBehavior(node, item);
  applyBoundsFromAbsoluteJson(node, item, isRoot ? null : { bounds: parentBounds });
  if (figmaType === "TEXT") {
    if ((await applyFinalJsonTextStyle(node, item, { bounds: parentBounds }, sourceContentMap)) === 0 && !item.fills) {
      node.fills = [{ type: "SOLID", color: { r: 0.05, g: 0.06, b: 0.08 } }];
    }
  } else {
    applyJsonVisualStyle(node, item);
  }
  parent.appendChild(node);

  const children = Array.isArray(item.children) ? item.children : [];
  for (const child of children) {
    await createNodeFromJsonItem(child, node, bounds, false, sourceContentMap);
  }
  return node;
}

async function drawJsonTreeBesideSelection(finalJson, sourceFrame, targetResolution, sourceContentMap) {
  if (!finalJson || typeof finalJson !== "object") {
    throw new Error("final_json must be an object.");
  }
  const rootBounds = jsonBounds(finalJson);
  const root = figma.createFrame();
  root.name = targetSizeName(targetResolution);
  root.x = sourceFrame.x + sourceFrame.width + BESIDE_FRAME_GAP;
  root.y = sourceFrame.y;
  root.layoutMode = "NONE";
  root.clipsContent = true;
  root.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  resizeNodeIfPossible(root, rootBounds.width, rootBounds.height);
  applyJsonVisualStyle(root, finalJson);
  figma.currentPage.appendChild(root);

  const children = Array.isArray(finalJson.children) ? finalJson.children : [];
  for (const child of children) {
    await createNodeFromJsonItem(child, root, rootBounds, false, sourceContentMap);
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
  clone.x = selectedFrame.x + selectedFrame.width + BESIDE_FRAME_GAP;
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

  function walk(item, parentJsonNode, isRoot) {
    if (!item || typeof item !== "object") return;
    const node = isRoot ? convertedFrame : resolve(item);
    const bounds = item.bounds && typeof item.bounds === "object" ? item.bounds : null;
    if (node) {
      if (!isRoot && bounds) {
        applyBoundsFromAbsoluteJson(node, item, parentJsonNode);
      } else if (isRoot && bounds && typeof bounds.width === "number" && typeof bounds.height === "number") {
        resizeNodeIfPossible(node, bounds.width, bounds.height);
      }
      if (item.name) node.name = String(item.name);
      applied++;
    } else if (item.id || item.path) {
      missing.push(String(item.id || item.path));
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) walk(child, item, false);
  }

  walk(predictedJson, null, true);
  return { applied, missing };
}

function findTextNodeBySemanticRole(root, role) {
  if (!root || !role || !("findAll" in root)) return null;
  const wanted = String(role).trim();
  const matches = root.findAll((node) => {
    if (!node || !("characters" in node)) return false;
    const directName = String(node.name || "").trim();
    let semanticName = "";
    try {
      semanticName = String(node.getPluginData("semanticName") || node.getPluginData("semanticRole") || "").trim();
    } catch (_e) {
      semanticName = "";
    }
    return directName === wanted || semanticName === wanted;
  });
  return matches.length ? matches[0] : null;
}

function resolveSourceTextNodeForJson(item, sourceNodeByOriginalId, sourcePathNodeMap, sourceFrame) {
  if (!item || typeof item !== "object") return null;
  const id = String(item.id || "").trim();
  const path = String(item.path || "").trim();
  if (id && sourceNodeByOriginalId && sourceNodeByOriginalId.has(id)) {
    const node = sourceNodeByOriginalId.get(id);
    if (node && "characters" in node) return node;
  }
  if (path && sourcePathNodeMap && sourcePathNodeMap.has(path)) {
    const node = sourcePathNodeMap.get(path);
    if (node && "characters" in node) return node;
  }
  return findTextNodeBySemanticRole(sourceFrame, semanticRoleName(item));
}

function loadUniformTextFont(node, role) {
  const fontName = node && node.fontName;
  if (fontName && typeof fontName === "object" && fontName.family && fontName.style) {
    return loadJsonFontName(fontName);
  }
  console.error("[TEXT_FONT_LOAD_FAILED]", role, "source text has mixed or invalid fontName", fontName);
  throw new Error(`Source text font unavailable for ${role}: mixed or invalid fontName.`);
}

async function cloneSourceTextIntoJsonParent(item, parentJsonNode, convertedFrame, resolveTarget, sourceNodeByOriginalId, sourcePathNodeMap, sourceFrame) {
  const role = semanticRoleName(item);
  const sourceText = resolveSourceTextNodeForJson(item, sourceNodeByOriginalId, sourcePathNodeMap, sourceFrame);
  if (!sourceText || !("characters" in sourceText)) {
    console.warn("[TEXT_CLONE_MISSING_SOURCE]", role, item && (item.id || item.path || item.name));
    return null;
  }

  const parentNode = parentJsonNode ? resolveTarget(parentJsonNode, false) : convertedFrame;
  if (!parentNode || typeof parentNode.appendChild !== "function") {
    console.warn("[TEXT_CLONE_MISSING_PARENT]", role, parentJsonNode && (parentJsonNode.id || parentJsonNode.path || parentJsonNode.name));
    return null;
  }

  const existing = resolveTarget(item, false);
  const insertIndex =
    existing && existing.parent && existing.parent.id === parentNode.id && Array.isArray(parentNode.children)
      ? parentNode.children.indexOf(existing)
      : parentNode.children.length;

  const node = sourceText.clone();
  try {
    node.setPluginData("originalNodeId", String(item.id || ""));
    node.setPluginData("semanticName", role);
  } catch (_e) {
    /* plugin data is best-effort */
  }
  node.name = role || String(item.name || sourceText.name || "text");

  if (insertIndex >= 0 && insertIndex < parentNode.children.length && typeof parentNode.insertChild === "function") {
    parentNode.insertChild(insertIndex, node);
  } else {
    parentNode.appendChild(node);
  }
  if (existing && existing.id !== node.id) {
    try {
      existing.remove();
    } catch (e) {
      console.warn("[TEXT_CLONE_REMOVE_OLD_FAILED]", role, e);
    }
  }

  await applyFinalJsonTextStyle(node, item, parentJsonNode, null);
  return node;
}

async function applyFinalJsonContentToClone(finalJson, convertedFrame, sourceFrame, sourceContentMap) {
  const { map: nodeByOriginalId } = collectClonedNodesByOriginalId(convertedFrame);
  const { map: sourceNodeByOriginalId } = sourceFrame ? collectClonedNodesByOriginalId(sourceFrame) : { map: new Map() };
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  const sourcePathNodeMap = sourceFrame ? buildPathNodeMap(sourceFrame) : new Map();
  let applied = 0;
  const missing = [];

  function resolve(item, isRoot) {
    if (!item || typeof item !== "object") return null;
    if (isRoot) return convertedFrame;
    const id = String(item.id || "").trim();
    const path = String(item.path || "").trim();
    if (id && nodeByOriginalId.has(id)) return nodeByOriginalId.get(id);
    if (path && pathNodeMap.has(path)) return pathNodeMap.get(path);
    return null;
  }

  async function walk(item, parentJsonNode, isRoot) {
    if (!item || typeof item !== "object") return;
    if (!isRoot && sourceFrame && shouldCloneSourceTextForRole(item)) {
      const cloned = await cloneSourceTextIntoJsonParent(
        item,
        parentJsonNode,
        convertedFrame,
        resolve,
        sourceNodeByOriginalId,
        sourcePathNodeMap,
        sourceFrame,
      );
      if (cloned) applied++;
      const children = Array.isArray(item.children) ? item.children : [];
      for (const child of children) {
        await walk(child, item, false);
      }
      return;
    }

    const node = resolve(item, isRoot);
    if (!node) {
      if (item.id || item.path) missing.push(String(item.id || item.path));
      return;
    }

    if (typeof item.visible === "boolean") {
      node.visible = item.visible;
      applied++;
    }
    if (typeof item.opacity === "number" && Number.isFinite(item.opacity)) {
      try {
        node.opacity = item.opacity;
        applied++;
      } catch (_e) {
        /* some nodes may reject opacity updates */
      }
    }
    applyJsonClippingBehavior(node, item);
    const isJsonTextNode = "characters" in item && "characters" in node;
    if (!isJsonTextNode) {
      applied += applyJsonVisualStyle(node, item);
      applied += applySourceImageContent(node, item, sourceContentMap);
    }

    if (isJsonTextNode) {
      try {
        applied += await applyFinalJsonTextStyle(node, item, parentJsonNode, sourceContentMap);
      } catch (e) {
        console.warn("applyFinalJsonContentToClone: text content apply failed", node && node.name, e);
      }
    }

    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) {
      await walk(child, item, false);
    }
  }

  await walk(finalJson, null, true);
  return { applied, missing };
}

async function applyAllFinalJsonTextStyles(finalJson, convertedFrame, sourceContentMap) {
  const { map: nodeByOriginalId } = collectClonedNodesByOriginalId(convertedFrame);
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  let applied = 0;
  const missing = [];

  function addTextCandidate(out, seen, node) {
    if (!node || !("characters" in node) || seen.has(node.id)) return;
    seen.add(node.id);
    out.push(node);
  }

  function resolveTextNodes(item, isRoot) {
    const out = [];
    const seen = new Set();
    if (!item || typeof item !== "object" || isRoot) return out;
    const id = String(item.id || "").trim();
    const path = String(item.path || "").trim();
    if (id && nodeByOriginalId.has(id)) {
      const node = nodeByOriginalId.get(id);
      addTextCandidate(out, seen, node);
    }
    if (path && pathNodeMap.has(path)) {
      const node = pathNodeMap.get(path);
      addTextCandidate(out, seen, node);
    }
    if (convertedFrame && typeof convertedFrame.findAll === "function") {
      const role = semanticRoleName(item);
      const matches = convertedFrame.findAll((node) => {
        if (!node || !("characters" in node)) return false;
        const nodeName = String(node.name || "").trim();
        let semanticName = "";
        let semanticRole = "";
        let originalId = "";
        try {
          semanticName = String(node.getPluginData("semanticName") || "").trim();
          semanticRole = String(node.getPluginData("semanticRole") || "").trim();
          originalId = String(node.getPluginData("originalNodeId") || "").trim();
        } catch (_e) {
          semanticName = "";
          semanticRole = "";
          originalId = "";
        }
        return (
          (id && originalId === id) ||
          (role && (nodeName === role || semanticName === role || semanticRole === role)) ||
          (item.name && nodeName === String(item.name).trim())
        );
      });
      for (const node of matches) addTextCandidate(out, seen, node);
    }
    return out;
  }

  async function walk(item, parentJsonNode, isRoot) {
    if (!item || typeof item !== "object") return;
    if ("characters" in item) {
      const nodes = resolveTextNodes(item, isRoot);
      if (nodes.length > 0) {
        for (const node of nodes) {
          try {
            applied += await applyFinalJsonTextStyle(node, item, parentJsonNode, sourceContentMap);
          } catch (e) {
            console.warn("[TEXT_JSON_APPLY_FAILED]", item && (item.name || item.id || item.path), e);
          }
        }
      } else if (!isRoot) {
        missing.push(String(item.name || item.id || item.path || "text"));
      }
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) {
      await walk(child, item, false);
    }
  }

  await walk(finalJson, null, true);
  return { applied, missing };
}

async function pruneDuplicateFinalTextNodes(finalJson, convertedFrame) {
  let removed = 0;
  const kept = new Set();
  const missing = [];

  function candidatesFor(item) {
    if (!convertedFrame || typeof convertedFrame.findAll !== "function") return [];
    const role = semanticRoleName(item);
    const id = String(item && item.id ? item.id : "").trim();
    const matches = convertedFrame.findAll((node) => {
      if (!node || !("characters" in node)) return false;
      const nodeName = String(node.name || "").trim();
      let semanticName = "";
      let semanticRole = "";
      let originalId = "";
      try {
        semanticName = String(node.getPluginData("semanticName") || "").trim();
        semanticRole = String(node.getPluginData("semanticRole") || "").trim();
        originalId = String(node.getPluginData("originalNodeId") || "").trim();
      } catch (_e) {
        semanticName = "";
        semanticRole = "";
        originalId = "";
      }
      return (
        (id && originalId === id) ||
        (role && (nodeName === role || semanticName === role || semanticRole === role))
      );
    });
    matches.sort((a, b) => {
      const aTagged = _textNodeMatchesOriginalId(a, id) ? 0 : 1;
      const bTagged = _textNodeMatchesOriginalId(b, id) ? 0 : 1;
      if (aTagged !== bTagged) return aTagged - bTagged;
      return String(a.id).localeCompare(String(b.id));
    });
    return matches;
  }

  function walk(item, isRoot) {
    if (!item || typeof item !== "object") return;
    if (!isRoot && "characters" in item) {
      const matches = candidatesFor(item).filter((node) => !kept.has(node.id));
      if (matches.length === 0) {
        missing.push(String(item.name || item.id || item.path || "text"));
      } else {
        const keeper = matches[0];
        kept.add(keeper.id);
        for (let i = 1; i < matches.length; i++) {
          const node = matches[i];
          try {
            node.remove();
            removed++;
          } catch (e) {
            try {
              node.visible = false;
              removed++;
            } catch (_hideError) {
              console.warn("[TEXT_DUPLICATE_REMOVE_FAILED]", item && item.name, node && node.id, e);
            }
          }
        }
      }
      return;
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) walk(child, false);
  }

  walk(finalJson, true);
  return { removed, missing };
}

function _textNodeMatchesOriginalId(node, originalId) {
  if (!node || !originalId) return false;
  try {
    return String(node.getPluginData("originalNodeId") || "").trim() === String(originalId).trim();
  } catch (_e) {
    return false;
  }
}

async function replaceAllFinalJsonTextNodes(finalJson, convertedFrame, sourceContentMap) {
  let replaced = 0;
  const missing = [];

  function rebuildIndexes() {
    return {
      byId: collectClonedNodesByOriginalId(convertedFrame).map,
      byPath: buildPathNodeMap(convertedFrame),
    };
  }

  function resolveAnyNode(item, isRoot, indexes) {
    if (!item || typeof item !== "object") return null;
    if (isRoot) return convertedFrame;
    const id = String(item.id || "").trim();
    const path = String(item.path || "").trim();
    if (id && indexes.byId.has(id)) return indexes.byId.get(id);
    if (path && indexes.byPath.has(path)) return indexes.byPath.get(path);
    const role = semanticRoleName(item);
    if (role) {
      const matches = convertedFrame.findAll((node) => String(node.name || "").trim() === role);
      if (matches.length > 0) return matches[0];
    }
    return null;
  }

  function removeStaleTextNodesForJsonItem(item, keepNode) {
    if (!convertedFrame || typeof convertedFrame.findAll !== "function") return 0;
    const role = semanticRoleName(item);
    const originalId = String(item && item.id ? item.id : "").trim();
    let removed = 0;
    const staleNodes = convertedFrame.findAll((node) => {
      if (!node || node.id === keepNode.id || !("characters" in node)) return false;
      let semanticName = "";
      let storedOriginalId = "";
      try {
        semanticName = String(node.getPluginData("semanticName") || node.getPluginData("semanticRole") || "").trim();
        storedOriginalId = String(node.getPluginData("originalNodeId") || "").trim();
      } catch (_e) {
        semanticName = "";
        storedOriginalId = "";
      }
      const nodeName = String(node.name || "").trim();
      return (role && (nodeName === role || semanticName === role)) || (originalId && storedOriginalId === originalId);
    });
    for (const node of staleNodes) {
      try {
        node.remove();
        removed++;
      } catch (e) {
        try {
          node.visible = false;
          removed++;
          console.warn("[TEXT_STALE_HIDE_USED]", role, node.id, e);
        } catch (hideError) {
          console.warn("[TEXT_STALE_REMOVE_FAILED]", role, node && node.id, hideError);
        }
      }
    }
    return removed;
  }

  async function replaceTextNode(item, parentJsonNode, indexes) {
    const existing = resolveAnyNode(item, false, indexes);
    const parentNode = parentJsonNode ? resolveAnyNode(parentJsonNode, false, indexes) : convertedFrame;
    if (!parentNode || typeof parentNode.appendChild !== "function") {
      missing.push(`${item.name || item.path || item.id}:parent`);
      return;
    }

    const role = semanticRoleName(item);
    const jsonFontName =
      role === "headline"
        ? await loadHeadlineBoldFontName(item.fontName)
        : item.fontName && typeof item.fontName === "object" && item.fontName.family && item.fontName.style
          ? { family: String(item.fontName.family), style: String(item.fontName.style) }
          : { family: "Inter", style: "Regular" };
    let creationFontName = jsonFontName || { family: "Inter", style: "Regular" };
    try {
      await figma.loadFontAsync(jsonFontName);
    } catch (e) {
      console.warn("[TEXT_REPLACE_JSON_FONT_LOAD_FAILED]", role, jsonFontName, e);
      const existingFont = existing && existing.fontName && typeof existing.fontName === "object" && existing.fontName.family && existing.fontName.style
        ? { family: String(existing.fontName.family), style: String(existing.fontName.style) }
        : { family: "Inter", style: "Regular" };
      creationFontName = existingFont;
      try {
        await figma.loadFontAsync(creationFontName);
      } catch (fallbackError) {
        console.warn("[TEXT_REPLACE_FALLBACK_FONT_LOAD_FAILED]", role, creationFontName, fallbackError);
        missing.push(`${role || item.path || item.id}:font`);
        return;
      }
    }

    const newNode = figma.createText();
    newNode.name = role || String(item.name || "text");
    try {
      newNode.setPluginData("originalNodeId", String(item.id || ""));
      newNode.setPluginData("semanticName", role || "");
    } catch (_e) {
      /* plugin data best-effort */
    }
    newNode.fontName = creationFontName;
    const sourceCharacters = resolveSourceCharactersForJsonText(item, sourceContentMap);
    newNode.characters = String(sourceCharacters != null ? sourceCharacters : (item.characters || ""));
    if ("textAutoResize" in newNode) newNode.textAutoResize = "NONE";

    let insertIndex = -1;
    if (existing && existing.parent && existing.parent.id === parentNode.id && "children" in parentNode) {
      insertIndex = parentNode.children.indexOf(existing);
    }
    if (insertIndex >= 0 && typeof parentNode.insertChild === "function") {
      parentNode.insertChild(insertIndex, newNode);
    } else {
      parentNode.appendChild(newNode);
    }

    await applyFinalJsonTextStyle(newNode, item, parentJsonNode, sourceContentMap);
    let staleRemoved = 0;
    if (existing && existing.id !== newNode.id) {
      try {
        existing.remove();
        staleRemoved++;
      } catch (e) {
        console.warn("[TEXT_REPLACE_REMOVE_OLD_FAILED]", role, e);
      }
    }
    staleRemoved += removeStaleTextNodesForJsonItem(item, newNode);
    console.log("[TEXT_NODE_REPLACED_FROM_JSON]", role, {
      oldId: existing && existing.id,
      newId: newNode.id,
      fontSize: newNode.fontSize,
      fontName: newNode.fontName,
      alignH: newNode.textAlignHorizontal,
      staleRemoved,
    });
    replaced++;
  }

  async function walk(item, parentJsonNode, isRoot) {
    if (!item || typeof item !== "object") return;
    if (!isRoot && "characters" in item) {
      await replaceTextNode(item, parentJsonNode, rebuildIndexes());
      return;
    }
    const children = Array.isArray(item.children) ? item.children : [];
    for (const child of children) {
      await walk(child, item, false);
    }
  }

  await walk(finalJson, null, true);
  return { replaced, missing };
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

function collectFrameNodesFromSelection(selection) {
  const out = [];
  for (const n of selection) {
    if (n && n.type === "FRAME") {
      out.push(n);
    }
  }
  return out;
}

function getSelectionInfo() {
  const selection = figma.currentPage.selection;

  if (selection.length === 0) {
    return { hasSelection: false };
  }

  const frames = collectFrameNodesFromSelection(selection);
  const base = {
    hasSelection: true,
    selectionCount: selection.length,
    frameCount: frames.length,
    frames: frames.map(function (node) {
      return {
        id: node.id,
        name: node.name,
        type: node.type,
        width: "width" in node ? Number(node.width.toFixed(2)) : null,
        height: "height" in node ? Number(node.height.toFixed(2)) : null,
      };
    }),
  };

  if (frames.length === 0) {
    const node = selection[0];
    return Object.assign({}, base, {
      isFrame: false,
      id: node.id,
      name: node.name,
      type: node.type,
      width: "width" in node ? Number(node.width.toFixed(2)) : null,
      height: "height" in node ? Number(node.height.toFixed(2)) : null,
    });
  }

  const node = frames[0];
  return Object.assign({}, base, {
    isFrame: true,
    id: node.id,
    name: node.name,
    type: node.type,
    width: "width" in node ? Number(node.width.toFixed(2)) : null,
    height: "height" in node ? Number(node.height.toFixed(2)) : null,
  });
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

/** UI iframe completes POST /figma/convert-semantic-json with browser FormData (same as `frontend/figma.html`). */
const semanticJsonUiFetchWaiters = new Map();

function fetchSemanticJsonThroughUi(payload) {
  const id = "sj_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 12);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (semanticJsonUiFetchWaiters.has(id)) {
        semanticJsonUiFetchWaiters.delete(id);
        reject(new Error("Semantic JSON request timed out waiting for UI (12 min)."));
      }
    }, 12 * 60 * 1000);
    semanticJsonUiFetchWaiters.set(id, { resolve, reject, timeout });
    figma.ui.postMessage(
      Object.assign(
        {
          type: "semantic-json-fetch",
          id,
        },
        payload,
      ),
    );
  });
}

figma.ui.onmessage = async (msg) => {
  if (msg.type === "semantic-json-fetch-result-chunk") {
    const entry = semanticJsonUiFetchWaiters.get(msg.id);
    if (!entry) return;
    const total = Math.max(1, Number(msg.total) || 1);
    const index = Math.max(0, Number(msg.index) || 0);
    if (!entry.chunks || entry.total !== total) {
      entry.chunks = new Array(total);
      entry.total = total;
    }
    entry.chunks[index] = String(msg.chunk || "");
    return;
  }

  if (msg.type === "semantic-json-fetch-result") {
    const entry = semanticJsonUiFetchWaiters.get(msg.id);
    if (!entry) return;
    clearTimeout(entry.timeout);
    semanticJsonUiFetchWaiters.delete(msg.id);
    let data = msg.data;
    if (msg.ok && msg.chunked) {
      try {
        const chunks = entry.chunks || [];
        data = JSON.parse(chunks.join(""));
      } catch (e) {
        entry.reject(new Error("Semantic JSON chunked response could not be parsed: " + String(e && e.message ? e.message : e)));
        return;
      }
    }
    if (msg.ok && data && typeof data === "object" && "semantic_json" in data) {
      entry.resolve(data);
    } else {
      entry.reject(new Error(msg.error || "Semantic JSON request failed."));
    }
    return;
  }

  if (msg.type === "semantic-json-fetch-status") {
    postStatus(String(msg.message || ""));
    return;
  }

  if (msg.type === "export-selected-frame-rich-json") {
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      postError("Select one or more frames (only FRAME nodes are exported).");
      figma.ui.postMessage({ type: "rich-json-export-result", ok: false });
      sendSelectionInfo();
      return;
    }

    const origSel = selection.slice();
    var richOk = 0;
    var richFail = 0;
    for (var ri = 0; ri < frames.length; ri++) {
      var richFrame = frames[ri];
      try {
        figma.currentPage.selection = [richFrame];
        postStatus(
          "Rich JSON export (" +
            String(ri + 1) +
            "/" +
            String(frames.length) +
            "): serializing " +
            richFrame.name +
            "...",
        );
        stampOriginalNodeIds(richFrame);
        const origin = getOrigin(richFrame);
        const richJson = serializeNode(richFrame, origin, "");
        richJson.templateId = "figma_plugin_rich_json_export";
        const jsonText = JSON.stringify(richJson, null, 2);
        const safeBase =
          `${richFrame.name || "figma-export"}-${Math.round(richFrame.width)}x${Math.round(richFrame.height)}`;
        figma.ui.postMessage({
          type: "rich-json-export-result",
          ok: true,
          jsonText: jsonText,
          fileName: safeBase,
          batch: frames.length > 1,
          batchIndex: ri,
          batchTotal: frames.length,
        });
        postStatus(
          "Rich JSON export (" +
            String(ri + 1) +
            "/" +
            String(frames.length) +
            "): ok — " +
            String(jsonText.length) +
            " chars.",
        );
        richOk++;
      } catch (err) {
        console.error("Rich JSON export failed:", err);
        richFail++;
        const em = String(err && err.message ? err.message : err);
        postStatus(
          "Rich JSON export (" +
            String(ri + 1) +
            "/" +
            String(frames.length) +
            ") skipped: " +
            richFrame.name +
            " — " +
            em,
        );
        figma.ui.postMessage({
          type: "rich-json-export-result",
          ok: false,
          batch: frames.length > 1,
          batchIndex: ri,
          batchTotal: frames.length,
          error: em,
          frameName: richFrame.name,
        });
      }
    }
    figma.currentPage.selection = origSel;
    sendSelectionInfo();
    postStatus("Rich JSON batch done: " + String(richOk) + " ok, " + String(richFail) + " failed.");
    if (richFail > 0 && richOk === 0) {
      postError("All rich JSON exports in the batch failed. See status above.");
    }
    return;
  }

  if (msg.type === "export-selected-frame-html-css") {
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      postError("Select one or more frames (only FRAME nodes are exported).");
      figma.ui.postMessage({ type: "html-css-export-result", ok: false });
      sendSelectionInfo();
      return;
    }

    const origSel = selection.slice();
    var htmlOk = 0;
    var htmlFail = 0;
    for (var hi = 0; hi < frames.length; hi++) {
      var selectedFrame = frames[hi];
      try {
        figma.currentPage.selection = [selectedFrame];
        postStatus(
          "HTML/CSS export (" +
            String(hi + 1) +
            "/" +
            String(frames.length) +
            "): serializing " +
            selectedFrame.name +
            "...",
        );
        const origin = getOrigin(selectedFrame);
        const rawJson = serializeNode(selectedFrame, origin, "");
        rawJson.templateId = "figma_plugin_html_css_export";
        postStatus(
          "HTML/CSS export (" + String(hi + 1) + "/" + String(frames.length) + "): exporting banner PNG...",
        );
        const bannerPngBytes = await exportFramePngBytes(selectedFrame);
        const bannerPngBase64 = uint8ToBase64(bannerPngBytes);
        postStatus(
          "HTML/CSS export (" +
            String(hi + 1) +
            "/" +
            String(frames.length) +
            "): exporting element assets...",
        );
        const elementAssets = await exportElementAssetsForHtml(
          selectedFrame,
          rawJson,
          MAX_ELEMENT_LAYER_PNGS,
        );
        const html = rawJsonToHtmlCss(rawJson, bannerPngBase64, elementAssets);
        const safeBase =
          `${selectedFrame.name || "figma-export"}-${Math.round(selectedFrame.width)}x${Math.round(selectedFrame.height)}`;
        figma.ui.postMessage({
          type: "html-css-export-result",
          ok: true,
          html: html,
          fileName: safeBase,
          batch: frames.length > 1,
          batchAppend: hi > 0,
          batchIndex: hi,
          batchTotal: frames.length,
        });
        postStatus(
          "HTML/CSS export (" +
            String(hi + 1) +
            "/" +
            String(frames.length) +
            "): ok — " +
            String(elementAssets.length) +
            " assets, " +
            String(html.length) +
            " chars.",
        );
        htmlOk++;
      } catch (err) {
        console.error("HTML/CSS export failed:", err);
        htmlFail++;
        const em = String(err && err.message ? err.message : err);
        postStatus(
          "HTML/CSS export (" +
            String(hi + 1) +
            "/" +
            String(frames.length) +
            ") skipped: " +
            selectedFrame.name +
            " — " +
            em,
        );
        figma.ui.postMessage({
          type: "html-css-export-result",
          ok: false,
          batch: frames.length > 1,
          batchAppend: hi > 0,
          batchIndex: hi,
          batchTotal: frames.length,
          error: em,
          frameName: selectedFrame.name,
        });
      }
    }
    figma.currentPage.selection = origSel;
    sendSelectionInfo();
    postStatus("HTML/CSS batch done: " + String(htmlOk) + " ok, " + String(htmlFail) + " failed.");
    if (htmlFail > 0 && htmlOk === 0) {
      postError("All HTML/CSS exports in the batch failed. See status above.");
    }
    return;
  }

  if (msg.type === "pipeline-target-json-selected-frame") {
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Select one or more frames to run the banner classify → retrieve → draw pipeline.");
      sendSelectionInfo();
      return;
    }

    const backendUrl = String(msg.backendUrl || "").trim();
    if (!backendUrl) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Backend URL is empty.");
      return;
    }

    const origSel = selection.slice();
    let pipelineOk = 0;
    let pipelineFail = 0;

    try {
      figma.ui.postMessage({ type: "pipeline-busy", busy: true });
      for (let pi = 0; pi < frames.length; pi++) {
        const selectedFrame = frames[pi];
        try {
          figma.currentPage.selection = [selectedFrame];
          const targetResolution = parseTargetSize(msg.targetSize, selectedFrame);
          postStatus(`Pipeline (${pi + 1}/${frames.length}): preparing source frame… ${selectedFrame.name}`);
          const stampedNodeCount = stampOriginalNodeIds(selectedFrame);

          postStatus("Pipeline: serializing selected frame JSON...");
          const origin = getOrigin(selectedFrame);
          const rawJson = serializeNode(selectedFrame, origin, "");
          rawJson.templateId = "figma_plugin_pipeline_source";
          const sourceContentMap = attachTargetOrientationToSourceContentMap(
            collectSourceContentMap(selectedFrame),
            targetResolution,
          );

          postStatus("Pipeline: exporting banner PNG for Qwen classification...");
          const pngBytes = await exportFramePngBytes(selectedFrame);
          const pngBase64 = uint8ToBase64(pngBytes);

          postStatus(
            `Pipeline (${pi + 1}/${frames.length}): backend classify → retrieve for ${targetResolution.width}×${targetResolution.height}...`,
          );
          const result = await callBannerRawTargetPipeline(
            backendUrl,
            pngBase64,
            rawJson,
            targetResolution,
          );

          if (result && result.supported === false && Number(result.category) === -1) {
            const unsupportedMsg =
              String(result.message || "").trim() || "This banner is not supported now in the plugin.";
            const runHint = result.run_id ? `\nRun: ${result.run_id}` : "";
            postStatus(`Pipeline (${pi + 1}/${frames.length}): ${unsupportedMsg}${runHint}`);
            figma.notify(
              `No result created — banner not in the 6 supported campaigns (Qwen class -1).\n` +
                `${selectedFrame.name}\n` +
                `Target: ${targetResolution.width}×${targetResolution.height}` +
                runHint +
                `\nTry "Layout Transformer V2" for a direct model prediction.`,
              { timeout: 8 },
            );
            pipelineFail++;
            continue;
          }

          const usedLtFallback =
            Number(result.category) === -1 &&
            String(result.message || "").indexOf("layout_transformer_v2") !== -1;

          const sourceFilterReport = filterFinalJsonToSourceElements(result.final_json, selectedFrame);
          const layoutJson = sourceFilterReport.json;
          const portraitTypoJsonReport = polishPortraitTypographyInJson(layoutJson, selectedFrame, targetResolution);

          let convertedFrame;
          let drawMode;
          let applySummary = null;
          postStatus("Pipeline: drawing backend target JSON beside selection...");
          convertedFrame = await drawJsonTreeBesideSelection(layoutJson, selectedFrame, targetResolution, sourceContentMap);
          applyJsonTreeNamesByPath(layoutJson, convertedFrame);
          removeEmptyZeroSizeNodes(convertedFrame);
          drawMode = "create_from_backend_final_json";

          const recon = applyFinalJsonCloneReconstruction(layoutJson, convertedFrame);
          removeEmptyZeroSizeNodes(convertedFrame);
          applyJsonTreeNamesByPath(layoutJson, convertedFrame);
          removeEmptyZeroSizeNodes(convertedFrame);
          const contentReport = await applyFinalJsonContentToClone(layoutJson, convertedFrame, selectedFrame, sourceContentMap);
          const namingReport = finalizeSemanticLayerNamesFromJson(layoutJson, convertedFrame);
          const exactNameReport = applyExactNodeNamesFromJson(layoutJson, convertedFrame);
          removeEmptyZeroSizeNodes(convertedFrame);
          applyFinalAbsoluteBoundsCorrection(layoutJson, convertedFrame);
          const finalTextReport = await replaceAllFinalJsonTextNodes(layoutJson, convertedFrame, sourceContentMap);
          applyFinalAbsoluteBoundsCorrection(layoutJson, convertedFrame);
          const finalTextStyleReport = await applyAllFinalJsonTextStyles(layoutJson, convertedFrame, sourceContentMap);
          const portraitTypoFrameReport = await polishPortraitTypographyOnFrame(
            convertedFrame,
            layoutJson,
            targetResolution,
            sourceContentMap,
          );
          const duplicateTextReport = await pruneDuplicateFinalTextNodes(layoutJson, convertedFrame);
          const sourcePruneReport = pruneFigmaNodesNotInSource(selectedFrame, convertedFrame);
          convertedFrame.name = targetSizeName(targetResolution);

          figma.currentPage.selection = [convertedFrame];
          figma.viewport.scrollAndZoomIntoView([selectedFrame, convertedFrame]);

          if (frames.length === 1) {
            figma.notify(
              `Target clone created.\n` +
                `Qwen class: ${result.category}${usedLtFallback ? " (LT v2 fallback)" : ""}\n` +
                `${usedLtFallback ? "Mode: layout_transformer_v2 direct\n" : `Guide: ${(result.selected_candidate && result.selected_candidate.name) || "unknown"}\n`}` +
                `Draw: ${drawMode}\n` +
                `Hierarchy sync: ${recon.hierarchyReport.reparentMoves} · Pruned: ${recon.pruneReport.removed}\n` +
                `Reorder: ${recon.reorderAfterPrune.moves} + ${recon.reorderAfterStray.moves} · Stamp: ${recon.stampReport.stamped}` +
                (recon.stampReport.mismatchWarn ? ` (${recon.stampReport.mismatchWarn} count mismatches)` : "") +
                `\nStray layers removed: ${recon.strayReport.removed}\n` +
                `Layout (abs→rel): ${recon.layoutReport.applied} (skipped ${recon.layoutReport.skipped || 0}, json ids ${recon.layoutReport.indexIds}, map ${recon.layoutReport.mapSize})\n` +
                `Bounds doc-fix: corrected ${recon.boundsFixReport.corrected}, child skips ${recon.boundsFixReport.skippedChildren}\n` +
                `Empty frames removed: ${recon.emptyReport.removed}\n` +
                `Content applied: ${contentReport.applied} · Final text replaced: ${finalTextReport.replaced} · Final styles: ${finalTextStyleReport.applied} · Text dupes: ${duplicateTextReport.removed}\n` +
                `Source-only JSON drops: ${sourceFilterReport.removedCount} · Source-only Figma pruned: ${sourcePruneReport.removed}\n` +
                `Semantic names: ${namingReport.renamed} (id map ${namingReport.mapped}) · Exact final rename: ${exactNameReport.renamed}\n` +
                `Selected stamped nodes: ${stampedNodeCount}\n` +
                (applySummary
                  ? `Scaled candidate nodes: ${applySummary.applied}\nScale: ${applySummary.scale_x.toFixed(3)} × ${applySummary.scale_y.toFixed(3)}`
                  : `Fallback drawn from JSON`),
              { timeout: 8 },
            );
          } else {
            figma.notify(
              `Pipeline ${pi + 1}/${frames.length}: clone ready for "${selectedFrame.name}".\n` +
                `Qwen class: ${result.category} · Mode: ${drawMode} · Content: ${contentReport.applied} · Source drops: ${sourceFilterReport.removedCount}`,
              { timeout: 4 },
            );
          }

          pipelineOk++;
        } catch (err) {
          pipelineFail++;
          console.error("Pipeline target JSON failed:", err);
          const shortMsg = String(err && err.message ? err.message : err);
          postStatus(`Pipeline (${pi + 1}/${frames.length}) skipped: ${selectedFrame.name} — ${shortMsg}`);
          if (err && err.message === "Failed to fetch") {
            postStatus(
              "Figma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. Match Backend URL, then reload the plugin.",
            );
          }
        }
      }

      figma.currentPage.selection = origSel;
      sendSelectionInfo();
      figma.ui.postMessage({ type: "done" });
      if (frames.length > 1) {
        figma.notify(`Pipeline batch complete: ${pipelineOk} ok, ${pipelineFail} failed.`, { timeout: 6 });
      }
      if (pipelineFail > 0 && pipelineOk === 0) {
        postError("Every frame in the pipeline batch failed. See status lines above.");
      } else if (pipelineFail > 0) {
        postError(`Pipeline finished with ${pipelineFail} failure(s); ${pipelineOk} succeeded.`);
      }
    } catch (err) {
      console.error("Pipeline batch failed:", err);
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

  if (msg.type === "layout-transformer-convert-selected-frame" || msg.type === "layout-transformer-v2-convert-selected-frame") {
    const isTransformerV2 = msg.type === "layout-transformer-v2-convert-selected-frame";
    const transformerLabel = isTransformerV2 ? "Layout Transformer V2" : "Layout Transformer";
    const transformerEndpoint = isTransformerV2 ? "/api/layout-transformer-v2" : "/api/layout-transformer";
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError(`Select one or more clean semantic frames (only FRAME nodes run ${transformerLabel}).`);
      sendSelectionInfo();
      return;
    }

    const backendUrl = String(msg.backendUrl || "").trim();
    if (!backendUrl) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Backend URL is empty.");
      return;
    }

    const origSel = selection.slice();
    let ok = 0;
    let fail = 0;

    try {
      figma.ui.postMessage({ type: "pipeline-busy", busy: true });
      for (let li = 0; li < frames.length; li++) {
        const selectedFrame = frames[li];
        try {
          figma.currentPage.selection = [selectedFrame];
          const targetResolution = parseTargetSize(msg.targetSize, selectedFrame);
          postStatus(`${transformerLabel} (${li + 1}/${frames.length}): stamping… ${selectedFrame.name}`);
          stampOriginalNodeIds(selectedFrame);

          postStatus(`${transformerLabel}: serializing selected semantic JSON…`);
          const origin = getOrigin(selectedFrame);
          const rawJson = serializeNode(selectedFrame, origin, "");
          rawJson.templateId = isTransformerV2 ? "figma_plugin_layout_transformer_v2" : "figma_plugin_layout_transformer";
          const sourceContentMap = attachTargetOrientationToSourceContentMap(
            collectSourceContentMap(selectedFrame),
            targetResolution,
          );

          postStatus(
            `${transformerLabel} (${li + 1}/${frames.length}): calling backend ${targetResolution.width}×${targetResolution.height}…`,
          );
          const result = await callLayoutTransformer(backendUrl, rawJson, targetResolution, transformerEndpoint);
          const sourceFilterReport = filterFinalJsonToSourceElements(result.final_json, selectedFrame);
          const layoutJson = sourceFilterReport.json;
          polishPortraitTypographyInJson(layoutJson, selectedFrame, targetResolution);

          postStatus(`${transformerLabel}: cloning predicted frame beside selection…`);
          const layoutClone = cloneFrameBesideSource(selectedFrame);
          applyJsonTreeNamesByPath(layoutJson, layoutClone);
          const recon = applyFinalJsonCloneReconstruction(layoutJson, layoutClone);
          applyJsonTreeNamesByPath(layoutJson, layoutClone);
          const contentReport = await applyFinalJsonContentToClone(layoutJson, layoutClone, selectedFrame, sourceContentMap);
          applyFinalAbsoluteBoundsCorrection(layoutJson, layoutClone);
          const namingReport = finalizeSemanticLayerNamesFromJson(layoutJson, layoutClone);
          applyExactNodeNamesFromJson(layoutJson, layoutClone);
          applyFinalAbsoluteBoundsCorrection(layoutJson, layoutClone);
          const finalTextReport = await replaceAllFinalJsonTextNodes(layoutJson, layoutClone, sourceContentMap);
          applyFinalAbsoluteBoundsCorrection(layoutJson, layoutClone);
          const finalTextStyleReport = await applyAllFinalJsonTextStyles(layoutJson, layoutClone, sourceContentMap);
          const portraitTypoFrameReport = await polishPortraitTypographyOnFrame(
            layoutClone,
            layoutJson,
            targetResolution,
            sourceContentMap,
          );
          const duplicateTextReport = await pruneDuplicateFinalTextNodes(layoutJson, layoutClone);
          const sourcePruneReport = pruneFigmaNodesNotInSource(selectedFrame, layoutClone);
          layoutClone.name = targetSizeName(targetResolution);

          figma.currentPage.selection = [layoutClone];
          figma.viewport.scrollAndZoomIntoView([selectedFrame, layoutClone]);

          if (frames.length === 1) {
            figma.notify(
              `${transformerLabel} clone ready.\n` +
                `Sync: ${recon.hierarchyReport.reparentMoves} · Prune: ${recon.pruneReport.removed} · Stray: ${recon.strayReport.removed}\n` +
                `Reorder: ${recon.reorderAfterPrune.moves}+${recon.reorderAfterStray.moves} · Stamp: ${recon.stampReport.stamped}` +
                (recon.stampReport.mismatchWarn ? ` (${recon.stampReport.mismatchWarn} mismatches)` : "") +
                `\nLayout: ${recon.layoutReport.applied} · Bounds fix: ${recon.boundsFixReport.corrected} · Empty removed: ${recon.emptyReport.removed}\n` +
                `Text/content: ${contentReport.applied} · Final text replaced: ${finalTextReport.replaced} · Final styles: ${finalTextStyleReport.applied} · Text dupes: ${duplicateTextReport.removed}\n` +
                `Source-only JSON drops: ${sourceFilterReport.removedCount} · Source-only Figma pruned: ${sourcePruneReport.removed} · Layers renamed: ${namingReport.renamed} (id map ${namingReport.mapped}).`,
              { timeout: 6 },
            );
          } else {
            figma.notify(`${transformerLabel} ${li + 1}/${frames.length}: done for "${selectedFrame.name}".`, {
              timeout: 4,
            });
          }
          ok++;
        } catch (err) {
          fail++;
          console.error(`${transformerLabel} convert failed:`, err);
          const shortMsg = String(err && err.message ? err.message : err);
          postStatus(`${transformerLabel} (${li + 1}/${frames.length}) skipped: ${selectedFrame.name} — ${shortMsg}`);
          if (err && err.message === "Failed to fetch") {
            postStatus(
              "Figma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. Match Backend URL, then reload the plugin.",
            );
          }
        }
      }

      figma.currentPage.selection = origSel;
      sendSelectionInfo();
      figma.ui.postMessage({ type: "done" });
      if (frames.length > 1) {
        figma.notify(`${transformerLabel} batch: ${ok} ok, ${fail} failed.`, { timeout: 5 });
      }
      if (fail > 0 && ok === 0) {
        postError(`Every frame in the ${transformerLabel} batch failed. See status above.`);
      } else if (fail > 0) {
        postError(`${transformerLabel} finished with ${fail} failure(s); ${ok} succeeded.`);
      }
    } catch (err) {
      console.error(`${transformerLabel} batch failed:`, err);
      let errMsg =
        err && err.stack ? err.message + "\n\n" + err.stack : String(err && err.message ? err.message : err);
      if (err && err.message === "Failed to fetch") {
        errMsg +=
          "\n\nFigma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. " +
          "Make sure Backend URL exactly matches the manifest, then reload the development plugin.";
      }
      postError(errMsg);
      sendSelectionInfo();
    } finally {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
    }
    return;
  }

  if (msg.type === "semantic-json-grid-selected-frame") {
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Select one or more frames (only FRAME nodes run semantic JSON).");
      sendSelectionInfo();
      return;
    }

    const backendUrl = String(msg.backendUrl || "").trim().replace(/\/+$/, "");
    if (!backendUrl) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Backend URL is empty.");
      return;
    }

    const maxNewTokens = Math.min(8192, Math.max(256, parseInt(String(msg.maxNewTokens || "2048"), 10) || 2048));
    const origSel = selection.slice();
    let semOk = 0;
    let semFail = 0;

    try {
      figma.ui.postMessage({
        type: "pipeline-busy",
        busy: true,
        overlayMessage: "Calling backend…",
      });
    for (let si = 0; si < frames.length; si++) {
      const selectedFrame = frames[si];
      try {
        figma.currentPage.selection = [selectedFrame];
        postStatus(`Semantic JSON (${si + 1}/${frames.length}): stamping… ${selectedFrame.name}`);
        stampOriginalNodeIds(selectedFrame);

        postStatus("Semantic JSON: serializing raw JSON…");
        const origin = getOrigin(selectedFrame);
        const rawJson = serializeNode(selectedFrame, origin, "");
        rawJson.templateId = "figma_plugin_semantic_grid";

        postStatus("Semantic JSON: exporting banner PNG…");
        const bannerBytes = await exportFramePngBytesMaxLongEdge(
          selectedFrame,
          SEMANTIC_BANNER_EXPORT_MAX_EDGE,
        );

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

        postStatus(
          `Semantic JSON (${si + 1}/${frames.length}): calling backend + Qwen via UI (may take several minutes)…`,
        );

        const jsonStr = JSON.stringify(rawJson);
        const bannerU8 =
          bannerBytes instanceof Uint8Array ? bannerBytes : new Uint8Array(bannerBytes);
        const atlasU8 =
          atlasPngBytes instanceof Uint8Array ? atlasPngBytes : new Uint8Array(atlasPngBytes);

        const data = await fetchSemanticJsonThroughUi({
          backendUrl,
          maxNewTokens,
          rawJsonText: jsonStr,
          bannerB64: uint8ToBase64(bannerU8),
          gridB64: uint8ToBase64(atlasU8),
        });

        const pretty = JSON.stringify(data.semantic_json, null, 2);

        postStatus("Semantic JSON: creating clone beside selection…");
        const semanticClone = cloneFrameBesideSource(selectedFrame);
        applyJsonTreeNamesByPath(data.semantic_json, semanticClone);
        removeEmptyZeroSizeNodes(semanticClone);
        postStatus("Semantic JSON: matching layer hierarchy to returned JSON…");
        const recon = applyFinalJsonCloneReconstruction(data.semantic_json, semanticClone);
        removeEmptyZeroSizeNodes(semanticClone);
        applyJsonTreeNamesByPath(data.semantic_json, semanticClone);
        removeEmptyZeroSizeNodes(semanticClone);
        const namingReport = finalizeSemanticLayerNamesFromJson(data.semantic_json, semanticClone);
        const exactNameReport = applyExactNodeNamesFromJson(data.semantic_json, semanticClone);
        removeEmptyZeroSizeNodes(semanticClone);
        const rootJsonLabel = String((data.semantic_json && data.semantic_json.name) || "").trim();
        semanticClone.name = buildSemanticCloneFrameTitle(selectedFrame.name, rootJsonLabel || "semantic");

        figma.currentPage.selection = [semanticClone];
        figma.viewport.scrollAndZoomIntoView([selectedFrame, semanticClone]);

        if (frames.length === 1) {
          figma.notify(
            `Semantic clone ready next to the original.\n` +
              `Sync: ${recon.hierarchyReport.reparentMoves} · Prune: ${recon.pruneReport.removed} · Stray: ${recon.strayReport.removed}\n` +
              `Reorder: ${recon.reorderAfterPrune.moves}+${recon.reorderAfterStray.moves} · Stamp: ${recon.stampReport.stamped}` +
              (recon.stampReport.mismatchWarn ? ` (${recon.stampReport.mismatchWarn} mismatches)` : "") +
              `\nLayout: ${recon.layoutReport.applied} (skipped ${recon.layoutReport.skipped || 0}) · Bounds fix: ${recon.boundsFixReport.corrected} · Empty removed: ${recon.emptyReport.removed}\n` +
              `Layers renamed: ${namingReport.renamed} (id map ${namingReport.mapped}) · Exact final rename: ${exactNameReport.renamed}.`,
            { timeout: 6 },
          );
        } else {
          figma.notify(
            `Semantic JSON ${si + 1}/${frames.length}: clone ready for "${selectedFrame.name}".`,
            { timeout: 4 },
          );
        }

        const fileBase = `${selectedFrame.name || "semantic"}-${Math.round(selectedFrame.width)}x${Math.round(selectedFrame.height)}`;
        figma.ui.postMessage({
          type: "semantic-json-result",
          ok: true,
          jsonText: pretty,
          fileName: fileBase,
          batch: frames.length > 1,
          batchAppend: si > 0,
          batchIndex: si,
          batchTotal: frames.length,
        });
        postStatus(
          `Semantic JSON (${si + 1}/${frames.length}): done. Clone "${semanticClone.name}" — ` +
            `${recon.hierarchyReport.reparentMoves} sync, ${recon.layoutReport.applied} layout, ` +
            `${recon.boundsFixReport.corrected} bounds-fix, ${namingReport.renamed} names, ${exactNameReport.renamed} exact overrides.`,
        );
        semOk++;
      } catch (err) {
        semFail++;
        console.error("Semantic JSON grid flow failed:", err);
        const shortMsg = err && err.message ? err.message : String(err);
        postStatus(`Semantic JSON (${si + 1}/${frames.length}) skipped: ${selectedFrame.name} — ${shortMsg}`);
        let errText =
          err && err.stack ? err.message + "\n\n" + err.stack : String(err && err.message ? err.message : err);
        if (err && err.message === "Failed to fetch") {
          errText +=
            "\n\nFigma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains. " +
            "Match Backend URL exactly, then reload the plugin.";
          postStatus(
            "Figma only allows requests to origins listed in manifest.json networkAccess.devAllowedDomains.",
          );
        }
        figma.ui.postMessage({
          type: "semantic-json-result",
          ok: false,
          error: shortMsg,
          batch: frames.length > 1,
          batchAppend: si > 0,
          batchIndex: si,
          batchTotal: frames.length,
          frameName: selectedFrame.name,
        });
      }
    }

    figma.currentPage.selection = origSel;
    postStatus(`Semantic JSON batch done: ${semOk} ok, ${semFail} failed.`);
    figma.ui.postMessage({ type: "done" });
    sendSelectionInfo();
    if (frames.length > 1) {
      figma.notify(`Semantic JSON batch: ${semOk} ok, ${semFail} failed.`, { timeout: 5 });
    }
    if (semFail > 0 && semOk === 0) {
      postError("All semantic JSON runs failed. See status / JSON panel.");
    } else if (semFail > 0) {
      postError(`${semFail} frame(s) failed; ${semOk} succeeded.`);
    }
    } finally {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
    }
    return;
  }
};

sendSelectionInfo();
