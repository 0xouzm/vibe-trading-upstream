import { initTheme } from "/theme.js";
import {
  DOCS_DEFAULT_LANG,
  DOCS_DEFAULT_PAGE,
  DOCS_LANGUAGES,
  DOCS_UI,
  DOCS_VERSIONS
} from "/docs/content.js?v=20260929";

const REPO = "HKUDS/Vibe-Trading";
const API = `https://api.github.com/repos/${REPO}`;
const STARS_CACHE_KEY = "vibetrading-github-stars";
const STARS_TTL_MS = 12 * 60 * 60 * 1000;
const LANG_KEY = "vibetrading-docs-lang";
const SITE = "https://vibetrading.wiki";

function stripTags(html) {
  return String(html).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
}

const pagesByLang = Object.fromEntries(
  Object.entries(DOCS_LANGUAGES).map(([lang, { structure }]) => [
    lang,
    structure.flatMap((group) =>
      group.pages.map((page) => ({
        ...page,
        group: group.label,
        searchText: [
          page.title,
          page.description,
          page.lead,
          ...page.sections.map((section) => `${section.title} ${stripTags(section.body)}`)
        ].join(" ").toLowerCase()
      }))
    )
  ])
);

function formatStarCount(n) {
  if (typeof n !== "number" || !Number.isFinite(n) || n < 0) return "--";
  if (n < 1000) return String(Math.round(n));
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}m`;
}

function initStars() {
  const el = document.getElementById("star-count");
  if (!el) return;

  let cached = null;
  try {
    cached = JSON.parse(localStorage.getItem(STARS_CACHE_KEY) || "null");
  } catch {
    cached = null;
  }
  if (cached && typeof cached.count === "number") el.textContent = formatStarCount(cached.count);
  if (cached && Date.now() - cached.at < STARS_TTL_MS) return;

  fetch(API, { headers: { Accept: "application/vnd.github+json" } })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
    .then((data) => {
      if (typeof data.stargazers_count !== "number") return;
      try {
        localStorage.setItem(STARS_CACHE_KEY, JSON.stringify({ count: data.stargazers_count, at: Date.now() }));
      } catch {
        /* ignore */
      }
      el.textContent = formatStarCount(data.stargazers_count);
    })
    .catch(() => {
      if (!cached) el.textContent = "--";
    });
}

function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function langForSegment(segment) {
  const match = Object.entries(DOCS_LANGUAGES).find(([, meta]) => meta.segment === segment);
  return match ? match[0] : DOCS_DEFAULT_LANG;
}

function routeParts() {
  const normalized = location.pathname.replace(/\/+$/, "");
  const match = normalized.match(/^\/docs\/([^/]+)\/(.+)$/);
  if (!match) {
    const bare = normalized.match(/^\/docs\/([^/]+)$/);
    const lang = bare ? langForSegment(bare[1]) : DOCS_DEFAULT_LANG;
    return { lang, pageId: DOCS_DEFAULT_PAGE, canonical: false };
  }
  const lang = langForSegment(match[1]);
  return { lang, pageId: match[2], canonical: match[1] === DOCS_LANGUAGES[lang].segment };
}

function canonicalPath(pageId, lang) {
  return `/docs/${DOCS_LANGUAGES[lang].segment}/${pageId}`;
}

function resolvePage(pageId, lang) {
  const pages = pagesByLang[lang];
  return pages.find((page) => page.id === pageId) || pages.find((page) => page.id === DOCS_DEFAULT_PAGE);
}

function storedLang() {
  try {
    const value = localStorage.getItem(LANG_KEY);
    return value && value in DOCS_LANGUAGES ? value : null;
  } catch {
    return null;
  }
}

function storeLang(lang) {
  try {
    localStorage.setItem(LANG_KEY, lang);
  } catch {
    /* ignore */
  }
}

function browserLang() {
  const first = (navigator.languages && navigator.languages[0]) || navigator.language || "";
  return first.toLowerCase().startsWith("zh") ? "zh" : DOCS_DEFAULT_LANG;
}

function setAlternate(hreflang, href) {
  let link = document.querySelector(`link[rel='alternate'][hreflang='${hreflang}']`);
  if (!link) {
    link = document.createElement("link");
    link.rel = "alternate";
    link.hreflang = hreflang;
    document.head.appendChild(link);
  }
  link.href = href;
}

function setMeta(page, lang) {
  const ui = DOCS_UI[lang];
  const title = `${page.title} - ${ui.siteTitle}`;
  const description = page.description || page.lead;
  document.title = title;
  document.querySelector('meta[name="description"]')?.setAttribute("content", description);
  document.querySelector('meta[property="og:title"]')?.setAttribute("content", title);
  document.querySelector('meta[property="og:description"]')?.setAttribute("content", description);
  document.querySelector('meta[property="og:site_name"]')?.setAttribute("content", ui.siteTitle);
  const canonical = `${SITE}${canonicalPath(page.id, lang)}`;
  document.querySelector('meta[property="og:url"]')?.setAttribute("content", canonical);
  document.querySelector("link[rel='canonical']")?.setAttribute("href", canonical);
  for (const [code, meta] of Object.entries(DOCS_LANGUAGES)) {
    setAlternate(meta.htmlLang, `${SITE}${canonicalPath(page.id, code)}`);
  }
  setAlternate("x-default", `${SITE}${canonicalPath(page.id, DOCS_DEFAULT_LANG)}`);
}

function applyChrome(lang, page) {
  const ui = DOCS_UI[lang];
  document.documentElement.lang = DOCS_LANGUAGES[lang].htmlLang;
  document.querySelectorAll("[data-ui]").forEach((el) => {
    const value = ui[el.getAttribute("data-ui")];
    if (typeof value === "string") el.textContent = value;
  });
  document.querySelectorAll("[data-ui-nav]").forEach((el) => {
    const value = ui.nav[el.getAttribute("data-ui-nav")];
    if (typeof value === "string") el.textContent = value;
  });
  document.getElementById("docs-search")?.setAttribute("placeholder", ui.searchPlaceholder);
  document.querySelector(".docs-sidebar")?.setAttribute("aria-label", ui.navLabel);
  document.querySelector(".docs-outline")?.setAttribute("aria-label", ui.onThisPage);

  const select = document.getElementById("version-select");
  if (select) {
    select.innerHTML = DOCS_VERSIONS.map((item) =>
      `<option value="${item.name}">${item.label[lang] || item.name}</option>`
    ).join("");
  }

  const toggle = document.getElementById("lang-toggle");
  if (toggle) {
    const other = Object.keys(DOCS_LANGUAGES).find((code) => code !== lang) || DOCS_DEFAULT_LANG;
    toggle.textContent = DOCS_LANGUAGES[other].label;
    toggle.setAttribute("href", canonicalPath(page.id, other));
    toggle.setAttribute("hreflang", DOCS_LANGUAGES[other].htmlLang);
    toggle.setAttribute("lang", DOCS_LANGUAGES[other].htmlLang);
    toggle.dataset.lang = other;
  }
}

function navLink(page, currentId) {
  const active = page.id === currentId ? "is-active" : "";
  return `<a class="${active}" data-doc-link="${page.id}">
    <span>${page.title}</span>
    <small>${page.description}</small>
  </a>`;
}

function renderNav(currentId, lang, filter = "") {
  const nav = document.getElementById("docs-nav");
  if (!nav) return;
  const q = filter.trim().toLowerCase();
  const pages = pagesByLang[lang];
  const groups = DOCS_LANGUAGES[lang].structure.map((group) => {
    const matches = pages.filter((page) => page.group === group.label && (!q || page.searchText.includes(q)));
    if (!matches.length) return "";
    return `<section>
      <h2>${group.label}</h2>
      ${matches.map((page) => navLink(page, currentId)).join("")}
    </section>`;
  }).join("");
  nav.innerHTML = groups || `<p class="empty-state">${DOCS_UI[lang].noMatch}</p>`;
}

function renderOutline(page) {
  const outline = document.getElementById("docs-outline");
  if (!outline) return;
  outline.innerHTML = page.sections.map((section) =>
    `<a href="#${section.id || slugify(section.title)}">${section.title}</a>`
  ).join("");
}

function renderArticle(page, lang) {
  const article = document.getElementById("docs-article");
  if (!article) return;

  const ui = DOCS_UI[lang];
  const pages = pagesByLang[lang];
  const index = pages.findIndex((candidate) => candidate.id === page.id);
  const previous = index > 0 ? pages[index - 1] : null;
  const next = index < pages.length - 1 ? pages[index + 1] : null;

  article.innerHTML = `
    <header class="doc-hero">
      <p class="eyebrow">${page.group}</p>
      <h1>${page.title}</h1>
      <p>${page.lead}</p>
    </header>
    ${page.sections.map((section) => `
      <section id="${section.id || slugify(section.title)}" class="doc-section">
        <h2>${section.title}</h2>
        ${section.body}
      </section>
    `).join("")}
    <footer class="doc-footer">
      ${previous ? `<a data-doc-link="${previous.id}"><span>${ui.previous}</span><strong>${previous.title}</strong></a>` : "<span></span>"}
      ${next ? `<a data-doc-link="${next.id}"><span>${ui.next}</span><strong>${next.title}</strong></a>` : "<span></span>"}
    </footer>
  `;
}

function navigate(path) {
  history.pushState({}, "", path);
  renderCurrent();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function bindDocLinks(lang, root = document) {
  root.querySelectorAll("[data-doc-link]").forEach((link) => {
    const pageId = link.getAttribute("data-doc-link");
    if (!pageId) return;
    link.setAttribute("href", canonicalPath(pageId, lang));
    link.onclick = (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) return;
      event.preventDefault();
      navigate(canonicalPath(pageId, lang));
    };
  });
}

function renderCurrent() {
  const { lang, pageId, canonical } = routeParts();
  const page = resolvePage(pageId, lang);

  if (!canonical || page.id !== pageId) {
    history.replaceState({}, "", canonicalPath(page.id, lang) + location.hash);
  }

  setMeta(page, lang);
  applyChrome(lang, page);
  renderNav(page.id, lang, document.getElementById("docs-search")?.value || "");
  renderArticle(page, lang);
  renderOutline(page);
  bindDocLinks(lang);
  document.getElementById("docs-article")?.focus({ preventScroll: true });
}

function initLanguage() {
  const { lang, pageId } = routeParts();
  const preferred = storedLang() || browserLang();
  if (lang === DOCS_DEFAULT_LANG && preferred !== DOCS_DEFAULT_LANG) {
    history.replaceState({}, "", canonicalPath(pageId, preferred) + location.hash);
  }

  document.getElementById("lang-toggle")?.addEventListener("click", (event) => {
    const target = event.currentTarget.dataset.lang;
    if (!target || !(target in DOCS_LANGUAGES)) return;
    storeLang(target);
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) return;
    event.preventDefault();
    navigate(event.currentTarget.getAttribute("href"));
  });
}

function initSearch() {
  const search = document.getElementById("docs-search");
  if (!search) return;
  search.addEventListener("input", () => {
    const { lang, pageId } = routeParts();
    renderNav(resolvePage(pageId, lang).id, lang, search.value);
    bindDocLinks(lang, document.getElementById("docs-nav") || document);
  });
}

function initHeaderScroll() {
  const header = document.getElementById("site-header");
  if (!header) return;
  const sync = () => header.classList.toggle("is-scrolled", window.scrollY > 10);
  sync();
  window.addEventListener("scroll", sync, { passive: true });
}

window.addEventListener("popstate", renderCurrent);

initTheme();
initStars();
initLanguage();
initSearch();
initHeaderScroll();
renderCurrent();
// The sections are rendered after load, so the browser could not scroll to
// a #section in the URL on its own.
if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
