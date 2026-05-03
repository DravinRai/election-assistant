/**
 * Election Education Assistant — Client-Side Logic
 * -------------------------------------------------
 * Handles:
 *   • Chat form submission & API communication
 *   • Conversation history management (multi-turn context)
 *   • Topic drawer toggle
 *   • Textarea auto-resize & character counter
 *   • Simple Markdown → HTML rendering
 *   • Accessibility announcements (screen-reader live region)
 *   • Google Translate integration (language selector)
 *   • Google TTS integration (Listen buttons)
 *   • Google Custom Search (News tab)
 *   • Google Maps (polling station search)
 *   • Google Analytics 4 event tracking
 *   • Tab navigation (Chat / News / Map)
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const chatMessages    = document.getElementById("chat-messages");
const chatForm        = document.getElementById("chat-form");
const userInput       = document.getElementById("user-input");
const btnSend         = document.getElementById("btn-send");
const btnTopics       = document.getElementById("btn-topics");
const topicDrawer     = document.getElementById("topic-drawer");
const typingIndicator = document.getElementById("typing-indicator");
const charCount       = document.getElementById("char-count");
const srAnnouncer     = document.getElementById("sr-announcer");
const langSelect      = document.getElementById("lang-select");

// News panel
const newsQuery       = document.getElementById("news-query");
const btnNewsSearch   = document.getElementById("btn-news-search");
const newsResults     = document.getElementById("news-results");

// Map panel
const mapAddress      = document.getElementById("map-address");
const btnFindPolling  = document.getElementById("btn-find-polling");
const mapIframe       = document.getElementById("map-iframe");

// Quiz panel
const btnStartQuiz    = document.getElementById("btn-start-quiz");
const quizContent     = document.getElementById("quiz-content");
const quizScoreBoard  = document.getElementById("quiz-score-board");
const lastScore       = document.getElementById("last-score");
const lastTotal       = document.getElementById("last-total");

// Timeline panel
const timelineCountry = document.getElementById("timeline-country");
const btnTimelineSearch = document.getElementById("btn-timeline-search");
const timelineContent = document.getElementById("timeline-content");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let conversationHistory = [];
let isGenerating = false;
let selectedLanguage = "en";
let currentAudio = null;
let sessionId = null;

// ---------------------------------------------------------------------------
// Init — create session & detect browser language
// ---------------------------------------------------------------------------
(async function init() {
    // Try to create a session for persistence
    try {
        const res = await fetch("/api/session", { method: "POST" });
        const data = await res.json();
        if (data.session_id) sessionId = data.session_id;
    } catch (_) { /* non-critical */ }

    // Auto-detect browser language and set selector if supported
    const browserLang = (navigator.language || "en").split("-")[0];
    if (langSelect.querySelector(`option[value="${browserLang}"]`)) {
        langSelect.value = browserLang;
        selectedLanguage = browserLang;
    }

    // GA4: page view
    trackEvent("page_view", { page_title: "Election Education Assistant" });
})();

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function announce(text) {
    srAnnouncer.textContent = text;
    setTimeout(() => { srAnnouncer.textContent = ""; }, 3000);
}

function scrollToBottom() {
    const container = chatMessages.closest(".chat-container");
    if (container) container.scrollTop = container.scrollHeight;
}

function renderMarkdown(md) {
    let html = md
        .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/^### (.+)$/gm, "<h4>$1</h4>")
        .replace(/^## (.+)$/gm, "<h3>$1</h3>")
        .replace(/^# (.+)$/gm, "<h2>$1</h2>")
        .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        .replace(/^[\-\*] (.+)$/gm, "<li>$1</li>")
        .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
        .replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>")
        .replace(/\n{2,}/g, "</p><p>")
        .replace(/\n/g, "<br>");
    html = `<p>${html}</p>`;
    html = html.replace(/<p>\s*<\/p>/g, "");
    return html;
}

function formatTopic(slug) {
    return slug.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
    const el = document.createElement("span");
    el.textContent = str;
    return el.innerHTML;
}

/** Track GA4 events (no-op if gtag not loaded) */
function trackEvent(eventName, params) {
    if (typeof gtag === "function") {
        gtag("event", eventName, params || {});
    }
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message user-message";
    el.innerHTML = `
        <div class="message-avatar" aria-hidden="true">You</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="message-author">You</span>
                <span class="message-badge">User</span>
            </div>
            <div class="message-content"><p>${escapeHtml(text)}</p></div>
        </div>`;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function addAssistantMessage(html, topic, rawText) {
    const el = document.createElement("div");
    el.className = "message assistant-message";
    const topicBadge = topic && topic !== "off_topic"
        ? `<span class="message-topic">${formatTopic(topic)}</span>`
        : "";

    // TTS listen button
    const listenBtn = `<button class="btn-listen" data-tts-text="${escapeHtml(rawText || '')}" aria-label="Listen to this response">
        <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>
        Listen
    </button>`;

    el.innerHTML = `
        <div class="message-avatar" aria-hidden="true">🏛️</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="message-author">ElectionEdu</span>
                <span class="message-badge">AI Assistant</span>
            </div>
            <div class="message-content">${html}${topicBadge}${listenBtn}</div>
        </div>`;
    chatMessages.appendChild(el);
    scrollToBottom();
    announce("New response from ElectionEdu assistant.");

    // Bind listen button
    const btn = el.querySelector(".btn-listen");
    if (btn) btn.addEventListener("click", () => handleTTS(btn));
}

function addErrorMessage(text) {
    const el = document.createElement("div");
    el.className = "message assistant-message";
    el.innerHTML = `
        <div class="message-avatar" aria-hidden="true">⚠️</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="message-author">System</span>
                <span class="message-badge" style="background:rgba(239,68,68,.15);color:var(--danger)">Error</span>
            </div>
            <div class="message-content" style="border-color:rgba(239,68,68,.25)"><p>${escapeHtml(text)}</p></div>
        </div>`;
    chatMessages.appendChild(el);
    scrollToBottom();
    announce("Error: " + text);
}

// ---------------------------------------------------------------------------
// Translation helper
// ---------------------------------------------------------------------------

async function translateIfNeeded(text) {
    if (selectedLanguage === "en") return text;
    try {
        const res = await fetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, target_language: selectedLanguage }),
        });
        const data = await res.json();
        if (data.success && data.translated_text) return data.translated_text;
    } catch (_) { /* fallback to original */ }
    return text;
}

// ---------------------------------------------------------------------------
// TTS handler
// ---------------------------------------------------------------------------

async function handleTTS(btn) {
    const text = btn.getAttribute("data-tts-text");
    if (!text) return;

    // Stop current audio if playing
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
        document.querySelectorAll(".btn-listen.playing").forEach(b => b.classList.remove("playing"));
        if (btn.classList.contains("playing")) { btn.classList.remove("playing"); return; }
    }

    btn.disabled = true;
    btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg> Loading…';

    try {
        const res = await fetch("/api/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, language: selectedLanguage }),
        });
        const data = await res.json();

        if (data.success && data.audio_base64) {
            const audio = new Audio(`data:audio/mp3;base64,${data.audio_base64}`);
            currentAudio = audio;
            btn.classList.add("playing");
            btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Playing…';
            audio.play();
            audio.onended = () => {
                btn.classList.remove("playing");
                btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg> Listen';
                currentAudio = null;
            };
            trackEvent("tts_play", { language: selectedLanguage });
        } else {
            btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg> Unavailable';
        }
    } catch (_) {
        btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg> Error';
    } finally {
        btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// API communication — Chat
// ---------------------------------------------------------------------------

async function sendMessage(text) {
    if (isGenerating) return;
    isGenerating = true;
    btnSend.disabled = true;
    typingIndicator.hidden = false;
    scrollToBottom();

    trackEvent("chat_message_sent", { message_length: text.length });

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                history: conversationHistory,
                session_id: sessionId,
            }),
        });

        const data = await res.json();
        typingIndicator.hidden = true;

        if (!res.ok || !data.success) {
            addErrorMessage(data.error || "Something went wrong. Please try again.");
            return;
        }

        // Update conversation history (multi-turn)
        conversationHistory.push({ role: "user", parts: [text] });
        conversationHistory.push({ role: "model", parts: [data.response] });

        if (conversationHistory.length > 40) {
            conversationHistory = conversationHistory.slice(-40);
        }

        // Translate if needed, then render
        const translated = await translateIfNeeded(data.response);
        addAssistantMessage(renderMarkdown(translated), data.topic, translated);

        trackEvent("chat_response", { topic: data.topic || "unknown" });

    } catch (err) {
        typingIndicator.hidden = true;
        addErrorMessage("Network error — please check your connection and try again.");
    } finally {
        isGenerating = false;
        btnSend.disabled = false;
        userInput.focus();
    }
}

// ---------------------------------------------------------------------------
// News search
// ---------------------------------------------------------------------------

async function searchNews(query) {
    if (!query) return;
    newsResults.innerHTML = '<div class="news-loading">Searching…</div>';
    trackEvent("news_search", { query });

    try {
        const res = await fetch(`/api/news?query=${encodeURIComponent(query)}&num=5`);
        const data = await res.json();

        if (!data.success || !data.results || data.results.length === 0) {
            newsResults.innerHTML = '<div class="news-placeholder"><p>No results found. Try a different query.</p></div>';
            return;
        }

        newsResults.innerHTML = data.results.map(r => `
            <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer" class="news-card">
                <div class="news-card-title">${escapeHtml(r.title)}</div>
                <div class="news-card-snippet">${escapeHtml(r.snippet)}</div>
                <div class="news-card-source">${escapeHtml(r.display_url || "Source")}</div>
            </a>
        `).join("");

    } catch (_) {
        newsResults.innerHTML = '<div class="news-placeholder"><p>Failed to fetch news. Please try again.</p></div>';
    }
}

// ---------------------------------------------------------------------------
// Map — polling station search
// ---------------------------------------------------------------------------

function searchPollingStation(address) {
    if (!address || !mapIframe) return;
    const encoded = encodeURIComponent(`polling stations near ${address}`);
    const currentSrc = mapIframe.src;
    // Extract API key from current src
    const keyMatch = currentSrc.match(/key=([^&]+)/);
    if (keyMatch) {
        mapIframe.src = `https://www.google.com/maps/embed/v1/search?key=${keyMatch[1]}&q=${encoded}&zoom=13`;
    }
    trackEvent("map_search", { address });
}

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        const isActive = btn.dataset.tab === tabName;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive);
    });
    document.querySelectorAll(".tab-panel").forEach(panel => {
        const isActive = panel.id === `panel-${tabName}`;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
    });
    trackEvent("tab_switch", { tab: tabName });
}

// ---------------------------------------------------------------------------
// Quiz logic
// ---------------------------------------------------------------------------

let currentQuizQuestion = null;
let quizScore = 0;
let questionsAnswered = 0;

async function loadNextQuizQuestion() {
    quizContent.innerHTML = '<div class="quiz-loading">Generating question…</div>';
    try {
        const res = await fetch("/api/quiz/question");
        const data = await res.json();
        if (data.success) {
            currentQuizQuestion = data;
            renderQuizQuestion(data);
        } else {
            quizContent.innerHTML = '<div class="quiz-error">Failed to load question.</div>';
        }
    } catch (_) {
        quizContent.innerHTML = '<div class="quiz-error">Network error.</div>';
    }
}

function renderQuizQuestion(q) {
    quizContent.innerHTML = `
        <div class="quiz-card">
            <div class="quiz-question">${escapeHtml(q.question)}</div>
            <div class="quiz-options">
                ${q.options.map((opt, i) => `
                    <button class="quiz-opt" data-index="${i}">${escapeHtml(opt)}</button>
                `).join("")}
            </div>
        </div>
    `;
    quizContent.querySelectorAll(".quiz-opt").forEach(btn => {
        btn.addEventListener("click", () => handleQuizAnswer(btn.textContent));
    });
}

async function handleQuizAnswer(selected) {
    const isCorrect = selected === currentQuizQuestion.correct_answer;
    if (isCorrect) quizScore++;
    questionsAnswered++;

    quizContent.innerHTML = `
        <div class="quiz-feedback ${isCorrect ? 'correct' : 'incorrect'}">
            <h3>${isCorrect ? '✅ Correct!' : '❌ Incorrect'}</h3>
            <p><strong>Explanation:</strong> ${renderMarkdown(currentQuizQuestion.explanation)}</p>
            <button id="btn-next-quiz" class="btn-primary">Next Question</button>
        </div>
    `;

    document.getElementById("btn-next-quiz").addEventListener("click", loadNextQuizQuestion);

    // Save score after each answer
    if (sessionId) {
        fetch(`/api/session/${sessionId}/quiz`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                score: quizScore,
                total: questionsAnswered,
                topic: currentQuizQuestion.topic || "general"
            })
        });
    }

    quizScoreBoard.hidden = false;
    lastScore.textContent = quizScore;
    lastTotal.textContent = questionsAnswered;
}

// ---------------------------------------------------------------------------
// Timeline logic
// ---------------------------------------------------------------------------

async function getTimeline(country) {
    if (!country) return;
    timelineContent.innerHTML = '<div class="timeline-loading">Fetching milestones…</div>';
    try {
        const res = await fetch("/api/timeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ country })
        });
        const data = await res.json();
        if (data.success && data.timeline) {
            renderTimeline(data);
        } else {
            timelineContent.innerHTML = '<div class="timeline-error">Failed to fetch timeline.</div>';
        }
    } catch (_) {
        timelineContent.innerHTML = '<div class="timeline-error">Network error.</div>';
    }
}

function renderTimeline(data) {
    timelineContent.innerHTML = `
        <div class="timeline-results">
            <h3>Election Milestones: ${escapeHtml(data.country)}</h3>
            <div class="timeline-list">
                ${data.timeline.map(item => `
                    <div class="timeline-item">
                        <div class="timeline-time">${escapeHtml(item.approximate_timeframe)}</div>
                        <div class="timeline-marker"></div>
                        <div class="timeline-info">
                            <div class="timeline-phase">${escapeHtml(item.phase)}</div>
                            <div class="timeline-desc">${escapeHtml(item.description)}</div>
                        </div>
                    </div>
                `).join("")}
            </div>
            <div class="timeline-summary">${renderMarkdown(data.summary)}</div>
        </div>
    `;
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

// Form submit
chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    addUserMessage(text);
    userInput.value = "";
    userInput.style.height = "auto";
    charCount.textContent = "0";
    sendMessage(text);
});

// Enter to send, Shift+Enter for newline
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

// Auto-resize textarea
userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
    charCount.textContent = userInput.value.length;
});

// Language selector
langSelect.addEventListener("change", () => {
    selectedLanguage = langSelect.value;
    trackEvent("language_change", { language: selectedLanguage });
    announce(`Language changed to ${langSelect.options[langSelect.selectedIndex].text}`);
});

// Topic drawer toggle
btnTopics.addEventListener("click", () => {
    const isOpen = topicDrawer.classList.toggle("open");
    btnTopics.setAttribute("aria-expanded", isOpen);
    if (isOpen) topicDrawer.removeAttribute("hidden");
    if (!isOpen) setTimeout(() => topicDrawer.setAttribute("hidden", ""), 400);
});

// Topic chip clicks
document.querySelectorAll(".topic-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        const query = chip.dataset.query;
        if (!query) return;
        switchTab("chat");
        addUserMessage(query);
        sendMessage(query);
        topicDrawer.classList.remove("open");
        btnTopics.setAttribute("aria-expanded", "false");
        setTimeout(() => topicDrawer.setAttribute("hidden", ""), 400);
        trackEvent("topic_click", { topic: query });
    });
});

// Tab buttons
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// News search
btnNewsSearch.addEventListener("click", () => searchNews(newsQuery.value.trim()));
newsQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); searchNews(newsQuery.value.trim()); }
});

// Map search
if (btnFindPolling) {
    btnFindPolling.addEventListener("click", () => searchPollingStation(mapAddress.value.trim()));
}
if (mapAddress) {
    mapAddress.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); searchPollingStation(mapAddress.value.trim()); }
    });
}

// Quiz start
if (btnStartQuiz) {
    btnStartQuiz.addEventListener("click", () => {
        quizScore = 0;
        questionsAnswered = 0;
        loadNextQuizQuestion();
    });
}

// Timeline search
if (btnTimelineSearch) {
    btnTimelineSearch.addEventListener("click", () => getTimeline(timelineCountry.value.trim()));
}
if (timelineCountry) {
    timelineCountry.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); getTimeline(timelineCountry.value.trim()); }
    });
}
