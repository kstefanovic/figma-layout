/**
 * Figma plugin main thread. Flows:
 * - ``POST …/pipeline/banner-raw-to-target-json-json`` — banner + raw JSON + target size → layout clone.
 * - ``POST …/layout-engine/convert`` — serialized frame JSON + target size → ``layout_engine.convert`` CP-SAT output; plugin clones beside the original and applies returned ``final_json``.
 * - ``POST …/figma/convert-semantic-json`` — banner + grid PNG + raw JSON → Qwen returns ``{names:{id:…}}`` merged server-side into full semantic JSON; plugin clones beside the original, reparents to match JSON hierarchy, then renames from that JSON.
 * - HTML/CSS export from serialized JSON + assets (local).
 */
figma.showUI(__html__, { width: 400, height: 760 });

/** Horizontal gap between the source frame and a sibling created by the plugin (px). */
const BESIDE_FRAME_GAP = 80;

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
 * Call before ``applyJsonTreeNamesByOriginalIds`` so the layer list matches backend hierarchy.
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
 * Re-stamp ``originalNodeId`` on the clone from ``jsonTree`` using **index-aligned** pairing:
 * ``jsonTree.children[i]`` ↔ ``cloneRoot.children[i]`` at every level.
 * Call only after ``pruneClonedNodesMissingFromFinalJson`` + ``reorderCloneChildrenPerFinalJson`` so
 * counts match and booleans / groups are not mis-mapped to sibling slots.
 */
function stampCloneOriginalIdsFromJson(jsonTree, cloneRoot) {
  let stamped = 0;
  let mismatchWarn = 0;

  function walk(jNode, fNode) {
    if (!jNode || typeof jNode !== "object" || !fNode) return;
    const jid = String(jNode.id || "").trim();
    if (jid) {
      try {
        fNode.setPluginData("originalNodeId", jid);
        stamped++;
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

  walk(jsonTree, cloneRoot);
  return { stamped, mismatchWarn };
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
      if (oid && !wanted.has(oid)) toRemove.push(c);
    }
    for (let r = 0; r < toRemove.length; r++) {
      const c = toRemove[r];
      const p = c.parent;
      if (p && typeof p.insertChild === "function" && "children" in c && Array.isArray(c.children)) {
        let insertIdx = p.children.indexOf(c);
        if (insertIdx < 0) insertIdx = p.children.length;
        const lift = [...c.children];
        for (let k = 0; k < lift.length; k++) {
          try {
            p.insertChild(insertIdx, lift[k]);
            insertIdx++;
          } catch (e) {
            console.warn("removeStray: lift failed", lift[k] && lift[k].id, e);
          }
        }
      }
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

  if (typeof jsonBounds.width === "number" && typeof jsonBounds.height === "number" && "resizeWithoutConstraints" in node) {
    try {
      node.resizeWithoutConstraints(Math.max(0.01, jsonBounds.width), Math.max(0.01, jsonBounds.height));
    } catch (_e) {
      /* text / etc. */
    }
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
  const hierarchyReport = syncCloneHierarchyToJsonTree(jsonTree, cloneRoot);
  const pruneReport = pruneClonedNodesMissingFromFinalJson(jsonTree, cloneRoot);
  const reorderAfterPrune = reorderCloneChildrenPerFinalJson(jsonTree, cloneRoot);
  const stampReport = stampCloneOriginalIdsFromJson(jsonTree, cloneRoot);
  const strayReport = removeStrayFigmaChildrenNotInJson(jsonTree, cloneRoot);
  const reorderAfterStray = reorderCloneChildrenPerFinalJson(jsonTree, cloneRoot);
  const layoutReport = applyFinalJsonAbsoluteLayout(jsonTree, cloneRoot);
  const emptyReport = removeEmptyFramesUnder(cloneRoot);
  const boundsFixReport = applyFinalAbsoluteBoundsCorrection(jsonTree, cloneRoot);
  return {
    hierarchyReport,
    pruneReport,
    reorderAfterPrune,
    stampReport,
    strayReport,
    reorderAfterStray,
    layoutReport,
    emptyReport,
    boundsFixReport,
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

  function walk(item, parentJsonAbs) {
    if (!item || typeof item !== "object") return;
    const abs = jsonBounds(item);
    const id = String(item.id || "").trim();
    const node = id ? byId.get(id) : null;

    if (node && cloneRoot && node.id === cloneRoot.id) {
      if (typeof abs.width === "number" && typeof abs.height === "number") {
        resizeNodeIfPossible(node, abs.width, abs.height);
        applied++;
      }
    } else if (node) {
      if (typeof abs.width === "number" && typeof abs.height === "number") {
        resizeNodeIfPossible(node, abs.width, abs.height);
      }
      const relX = abs.x - parentJsonAbs.x;
      const relY = abs.y - parentJsonAbs.y;
      node.x = relX;
      node.y = relY;
      applied++;
    } else if (id) {
      skipped++;
    }

    const passToChildren = id ? abs : parentJsonAbs;
    const kids = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < kids.length; i++) {
      walk(kids[i], passToChildren);
    }
  }

  const rootAbs = jsonBounds(jsonTree);
  walk(jsonTree, rootAbs);

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
      const p = c.parent;
      if (p && typeof p.insertChild === "function" && "children" in c && Array.isArray(c.children)) {
        let insertIdx = p.children.indexOf(c);
        if (insertIdx < 0) insertIdx = p.children.length;
        const lift = [...c.children];
        for (let k = 0; k < lift.length; k++) {
          try {
            p.insertChild(insertIdx, lift[k]);
            insertIdx++;
          } catch (e) {
            console.warn("prune: failed to lift child before removing orphan wrapper", lift[k] && lift[k].id, e);
          }
        }
      }
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
 * Remove empty FRAME/GROUP nodes under ``root`` (e.g. raw wrappers left after reparent + prune).
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
      if (!node || !("children" in node) || !Array.isArray(node.children)) return;
      for (let i = 0; i < node.children.length; i++) {
        collectEmpty(node.children[i]);
      }
      const t = node.type;
      if (node !== root && (t === "FRAME" || t === "GROUP") && node.children.length === 0) {
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
 * Walk a semantic / ``final_json`` tree and rename nodes in ``cloneRoot`` by serialized Figma ``id``,
 * using ``originalNodeId`` plugin data (``collectClonedNodesByOriginalId``).
 */
function applyJsonTreeNamesByOriginalIds(jsonTree, cloneRoot) {
  const { map: byId, mapped } = collectClonedNodesByOriginalId(cloneRoot);
  let renamed = 0;
  const missing = [];

  function walk(item) {
    if (!item || typeof item !== "object") return;
    const id = String(item.id || "").trim();
    const nm = String(item.name || "").trim();
    if (id && nm) {
      const node = byId.get(id);
      if (node) {
        if (setSemanticName(node, nm)) renamed++;
      } else {
        missing.push(id);
      }
    }
    const kids = Array.isArray(item.children) ? item.children : [];
    for (let i = 0; i < kids.length; i++) walk(kids[i]);
  }

  walk(jsonTree);
  return { renamed, missing, mapped };
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

async function callLayoutEngineConvert(backendUrl, rawJson, targetResolution) {
  const url = String(backendUrl || "").trim().replace(/\/+$/, "");
  if (!url) throw new Error("Backend URL is empty.");
  const response = await fetch(url + "/layout-engine/convert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      raw_json: rawJson,
      target_width: targetResolution.width,
      target_height: targetResolution.height,
      target_resolution: `${targetResolution.width}x${targetResolution.height}`,
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
    throw new Error(`Layout engine backend failed: ${detail}`);
  }
  if (!data || typeof data !== "object" || !data.final_json) {
    throw new Error("Layout engine returned invalid JSON or missing final_json.");
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
    // ``bounds`` / ``parentBounds`` are absolute in banner-root space (same as backend ``final_json``).
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
  root.x = sourceFrame.x + sourceFrame.width + BESIDE_FRAME_GAP;
  root.y = sourceFrame.y;
  root.layoutMode = "NONE";
  root.clipsContent = false;
  root.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  resizeNodeIfPossible(root, rootBounds.width, rootBounds.height);
  figma.currentPage.appendChild(root);

  const children = Array.isArray(finalJson.children) ? finalJson.children : [];
  for (const child of children) {
    await createNodeFromJsonItem(child, root, rootBounds, false);
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

figma.ui.onmessage = async (msg) => {
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
      postError("Select one or more frames (only FRAME nodes run the pipeline).");
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
          postStatus(`Pipeline (${pi + 1}/${frames.length}): stamping… ${selectedFrame.name}`);
          const stampedNodeCount = stampOriginalNodeIds(selectedFrame);

          postStatus("Pipeline: serializing selected Figma design...");
          const origin = getOrigin(selectedFrame);
          const rawJson = serializeNode(selectedFrame, origin, "");
          rawJson.templateId = "figma_plugin_pipeline_source";

          postStatus("Pipeline: exporting banner PNG...");
          const pngBytes = await exportFramePngBytes(selectedFrame);
          const pngBase64 = uint8ToBase64(pngBytes);

          postStatus(
            `Pipeline (${pi + 1}/${frames.length}): calling backend for ${targetResolution.width}×${targetResolution.height} target JSON...`,
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

          const rootJsonLabel = String((result.final_json && result.final_json.name) || "").trim();
          const recon = applyFinalJsonCloneReconstruction(result.final_json, convertedFrame);
          const namingReport = applyJsonTreeNamesByOriginalIds(result.final_json, convertedFrame);
          convertedFrame.name = buildSemanticCloneFrameTitle(selectedFrame.name, rootJsonLabel || "layout");

          figma.currentPage.selection = [convertedFrame];
          figma.viewport.scrollAndZoomIntoView([selectedFrame, convertedFrame]);

          if (frames.length === 1) {
            figma.notify(
              `Target clone created.\n` +
                `Qwen class: ${result.category}\n` +
                `Guide: ${selectedGuide.name || "unknown"}\n` +
                `Mode: ${drawMode}\n` +
                `Hierarchy sync: ${recon.hierarchyReport.reparentMoves} · Pruned: ${recon.pruneReport.removed}\n` +
                `Reorder: ${recon.reorderAfterPrune.moves} + ${recon.reorderAfterStray.moves} · Stamp: ${recon.stampReport.stamped}` +
                (recon.stampReport.mismatchWarn ? ` (${recon.stampReport.mismatchWarn} count mismatches)` : "") +
                `\nStray layers removed: ${recon.strayReport.removed}\n` +
                `Layout (abs→rel): ${recon.layoutReport.applied} (skipped ${recon.layoutReport.skipped || 0}, json ids ${recon.layoutReport.indexIds}, map ${recon.layoutReport.mapSize})\n` +
                `Bounds doc-fix: corrected ${recon.boundsFixReport.corrected}, child skips ${recon.boundsFixReport.skippedChildren}\n` +
                `Empty frames removed: ${recon.emptyReport.removed}\n` +
                `Semantic names: ${namingReport.renamed} (id map ${namingReport.mapped})\n` +
                `Selected stamped nodes: ${stampedNodeCount}\n` +
                (applySummary
                  ? `Scaled candidate nodes: ${applySummary.applied}\nScale: ${applySummary.scale_x.toFixed(3)} × ${applySummary.scale_y.toFixed(3)}`
                  : `Fallback drawn from JSON`),
              { timeout: 8 },
            );
          } else {
            figma.notify(
              `Pipeline ${pi + 1}/${frames.length}: clone ready for "${selectedFrame.name}".\n` +
                `Qwen class: ${result.category} · Mode: ${drawMode}`,
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

  if (msg.type === "layout-engine-convert-selected-frame") {
    const selection = figma.currentPage.selection;
    const frames = collectFrameNodesFromSelection(selection);
    if (frames.length === 0) {
      figma.ui.postMessage({ type: "pipeline-busy", busy: false });
      postError("Select one or more frames (only FRAME nodes run layout_engine).");
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
          postStatus(`Layout engine (${li + 1}/${frames.length}): stamping… ${selectedFrame.name}`);
          stampOriginalNodeIds(selectedFrame);

          postStatus("Layout engine: serializing frame JSON…");
          const origin = getOrigin(selectedFrame);
          const rawJson = serializeNode(selectedFrame, origin, "");
          rawJson.templateId = "figma_plugin_layout_engine";

          postStatus(
            `Layout engine (${li + 1}/${frames.length}): calling backend ${targetResolution.width}×${targetResolution.height}…`,
          );
          const result = await callLayoutEngineConvert(backendUrl, rawJson, targetResolution);
          const finalJson = result.final_json;

          postStatus("Layout engine: cloning frame beside selection…");
          const layoutClone = cloneFrameBesideSource(selectedFrame);
          const recon = applyFinalJsonCloneReconstruction(finalJson, layoutClone);
          const namingReport = applyJsonTreeNamesByOriginalIds(finalJson, layoutClone);
          const rootJsonLabel = String((finalJson && finalJson.name) || "").trim();
          layoutClone.name = buildSemanticCloneFrameTitle(
            selectedFrame.name,
            rootJsonLabel || targetSizeName(targetResolution),
          );

          figma.currentPage.selection = [layoutClone];
          figma.viewport.scrollAndZoomIntoView([selectedFrame, layoutClone]);

          if (frames.length === 1) {
            figma.notify(
              `Layout engine clone ready.\n` +
                `Sync: ${recon.hierarchyReport.reparentMoves} · Prune: ${recon.pruneReport.removed} · Stray: ${recon.strayReport.removed}\n` +
                `Reorder: ${recon.reorderAfterPrune.moves}+${recon.reorderAfterStray.moves} · Stamp: ${recon.stampReport.stamped}` +
                (recon.stampReport.mismatchWarn ? ` (${recon.stampReport.mismatchWarn} mismatches)` : "") +
                `\nLayout: ${recon.layoutReport.applied} · Bounds fix: ${recon.boundsFixReport.corrected} · Empty removed: ${recon.emptyReport.removed}\n` +
                `Layers renamed: ${namingReport.renamed} (id map ${namingReport.mapped}).`,
              { timeout: 6 },
            );
          } else {
            figma.notify(`Layout engine ${li + 1}/${frames.length}: done for "${selectedFrame.name}".`, {
              timeout: 4,
            });
          }
          ok++;
        } catch (err) {
          fail++;
          console.error("Layout engine convert failed:", err);
          const shortMsg = String(err && err.message ? err.message : err);
          postStatus(`Layout engine (${li + 1}/${frames.length}) skipped: ${selectedFrame.name} — ${shortMsg}`);
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
        figma.notify(`Layout engine batch: ${ok} ok, ${fail} failed.`, { timeout: 5 });
      }
      if (fail > 0 && ok === 0) {
        postError("Every frame in the layout_engine batch failed. See status above.");
      } else if (fail > 0) {
        postError(`Layout engine finished with ${fail} failure(s); ${ok} succeeded.`);
      }
    } catch (err) {
      console.error("Layout engine batch failed:", err);
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
      postError("Select one or more frames (only FRAME nodes run semantic JSON).");
      sendSelectionInfo();
      return;
    }

    const backendUrl = String(msg.backendUrl || "").trim().replace(/\/+$/, "");
    if (!backendUrl) {
      postError("Backend URL is empty.");
      return;
    }

    const maxNewTokens = Math.min(8192, Math.max(256, parseInt(String(msg.maxNewTokens || "2048"), 10) || 2048));
    const origSel = selection.slice();
    let semOk = 0;
    let semFail = 0;

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

        const requestUrl = `${backendUrl}/figma/convert-semantic-json`;
        postStatus(
          `Semantic JSON (${si + 1}/${frames.length}): calling backend + Qwen (may take several minutes)…`,
        );

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

        postStatus("Semantic JSON: creating clone beside selection…");
        const semanticClone = cloneFrameBesideSource(selectedFrame);
        postStatus("Semantic JSON: matching layer hierarchy to returned JSON…");
        const recon = applyFinalJsonCloneReconstruction(data.semantic_json, semanticClone);
        const namingReport = applyJsonTreeNamesByOriginalIds(data.semantic_json, semanticClone);
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
              `Layers renamed: ${namingReport.renamed} (id map ${namingReport.mapped}).`,
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
            `${recon.boundsFixReport.corrected} bounds-fix, ${namingReport.renamed} names.`,
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
    return;
  }
};

sendSelectionInfo();