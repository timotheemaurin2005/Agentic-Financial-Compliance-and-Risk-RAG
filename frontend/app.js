/* ═══════════════════════════════════════════════════════════
   FOMC Analyst — Interactive Effects & App Logic
   ═══════════════════════════════════════════════════════════ */

// ── State ──
let currentPage = 'landing';

// ── Smooth cursor tracking (spring/lerp for jelly effect) ──
const cursor = { x: 0, y: 0 };
const jellyPos = { x: 0, y: 0 };
const LERP_FACTOR = 0.06; // Lower = more elastic lag

function lerp(start, end, factor) {
  return start + (end - start) * factor;
}

function updateJellyLamp() {
  jellyPos.x = lerp(jellyPos.x, cursor.x, LERP_FACTOR);
  jellyPos.y = lerp(jellyPos.y, cursor.y, LERP_FACTOR);

  const hero = document.getElementById('hero');
  if (hero) {
    hero.style.setProperty('--mx', `${jellyPos.x}px`);
    hero.style.setProperty('--my', `${jellyPos.y}px`);
  }

  requestAnimationFrame(updateJellyLamp);
}

// Start the animation loop
requestAnimationFrame(updateJellyLamp);

// ── Hero mouse tracking ──
document.addEventListener('mousemove', (e) => {
  const hero = document.getElementById('hero');
  if (!hero) return;

  const rect = hero.getBoundingClientRect();
  cursor.x = e.clientX - rect.left;
  cursor.y = e.clientY - rect.top;
});

// ── Federal Funds Rate Chart — Scroll-triggered line draw ──
function initChartAnimation() {
  const chartWrapper = document.getElementById('chart-wrapper');
  if (!chartWrapper) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Animate the line
          const rateLine = document.getElementById('rate-line');
          const rateArea = document.getElementById('rate-area');

          if (rateLine) rateLine.classList.add('animate');
          if (rateArea) rateArea.classList.add('animate');

          // Animate data dots with staggered delay
          const dots = document.querySelectorAll('.data-dot');
          dots.forEach((dot, i) => {
            setTimeout(() => {
              dot.classList.add('animate');
            }, 300 + i * 400);
          });

          // Show rate value labels
          const rateValues = document.querySelectorAll('.rate-value');
          rateValues.forEach((val, i) => {
            setTimeout(() => {
              val.style.opacity = '1';
            }, 600 + i * 400);
          });

          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.3 }
  );

  observer.observe(chartWrapper);
}

// ── Scroll reveal animation ──
function initScrollReveal() {
  const reveals = document.querySelectorAll('.feature-card, .arch-node, .footer-title, .tech-stack');

  reveals.forEach((el) => el.classList.add('reveal'));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
  );

  reveals.forEach((el) => observer.observe(el));
}

// ── Stagger feature cards ──
function initCardStagger() {
  const cards = document.querySelectorAll('.feature-card');
  cards.forEach((card, i) => {
    card.style.transitionDelay = `${i * 0.12}s`;
  });
}

// ── Architecture node stagger ──
function initArchStagger() {
  const nodes = document.querySelectorAll('.arch-node');
  nodes.forEach((node, i) => {
    node.style.transitionDelay = `${i * 0.1}s`;
  });

  // Stagger connector pulse animations
  const pulses = document.querySelectorAll('.connector-pulse');
  pulses.forEach((pulse, i) => {
    pulse.style.animationDelay = `${i * 0.4}s`;
  });
}

// ── Navigation ──
function navigateToAnalyst() {
  const landing = document.getElementById('landing-page');
  const analyst = document.getElementById('analyst-page');

  if (landing && analyst) {
    landing.style.display = 'none';
    analyst.style.display = 'flex';
    currentPage = 'analyst';
    window.history.pushState({}, '', '/analyst');
  }
}

function navigateToLanding() {
  const landing = document.getElementById('landing-page');
  const analyst = document.getElementById('analyst-page');

  if (landing && analyst) {
    analyst.style.display = 'none';
    landing.style.display = 'block';
    currentPage = 'landing';
    window.history.pushState({}, '', '/');
  }
}

// Handle browser back/forward
window.addEventListener('popstate', () => {
  if (window.location.pathname === '/analyst') {
    navigateToAnalyst();
  } else {
    navigateToLanding();
  }
});

// ── Chat Interface — wired to FastAPI backend ──
async function sendMessage() {
  const input = document.getElementById('chat-input');
  const messages = document.getElementById('chat-messages');
  const contextPanel = document.getElementById('context-panel');
  const sendBtn = document.getElementById('send-btn');

  if (!input || !messages) return;

  const text = input.value.trim();
  if (!text) return;

  // Remove welcome message
  const welcome = messages.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  // Add user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  userMsg.innerHTML = `
    <div class="message-bubble">${escapeHtml(text)}</div>
    <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
  `;
  messages.appendChild(userMsg);

  input.value = '';
  input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;
  messages.scrollTop = messages.scrollHeight;

  // Add typing indicator
  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-message assistant';
  typingDiv.id = 'typing-indicator';
  typingDiv.innerHTML = `
    <div class="message-bubble">
      <span class="typing-status" id="typing-status">Connecting...</span>
      <span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>
    </div>
  `;
  messages.appendChild(typingDiv);
  messages.scrollTop = messages.scrollHeight;

  // Clear context panel
  if (contextPanel) {
    contextPanel.innerHTML = '<div class="context-empty"><p>Retrieving context...</p></div>';
  }

  try {
    const response = await fetch('/api/stream/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = '';
    let bubbleCreated = false;
    let assistantMsg = null;
    let bubbleEl = null;

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let newlineIndex;
      while ((newlineIndex = buffer.indexOf('\n\n')) >= 0) {
        // Extract a complete event chunk
        const eventChunk = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 2);

        if (!eventChunk) continue;

        // An event chunk might be multi-line, but the payload starts with 'data: '
        const lines = eventChunk.split('\n');
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;

          try {
            const event = JSON.parse(payload);

            if (event.type === 'status') {
              const statusEl = document.getElementById('typing-status');
              if (statusEl) statusEl.textContent = event.status;
            }

            if (event.type === 'token') {
              if (!bubbleCreated) {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
                assistantMsg = document.createElement('div');
                assistantMsg.className = 'chat-message assistant';
                assistantMsg.innerHTML = `
                  <div class="message-bubble"></div>
                  <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                `;
                messages.appendChild(assistantMsg);
                bubbleEl = assistantMsg.querySelector('.message-bubble');
                bubbleCreated = true;
              }
              assistantText += event.token;
              if (bubbleEl) bubbleEl.textContent = assistantText;
              messages.scrollTop = messages.scrollHeight;
            }

            if (event.type === 'final') {
              if (!bubbleCreated) {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
                assistantMsg = document.createElement('div');
                assistantMsg.className = 'chat-message assistant';
                assistantMsg.innerHTML = `
                  <div class="message-bubble">${escapeHtml(event.answer || 'No answer generated.')}</div>
                  <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                `;
                messages.appendChild(assistantMsg);
              }

              if (contextPanel && event.sources && event.sources.length > 0) {
                contextPanel.innerHTML = event.sources.map(src => `
                  <div class="context-chunk">
                    <div class="chunk-source">
                      <span class="chunk-badge">${escapeHtml(src.doc_type || 'DOC')}</span>
                      <span class="chunk-label">${escapeHtml(src.meeting_date || src.source || 'Unknown')}</span>
                    </div>
                    <p class="chunk-text">"${escapeHtml((src.text || src.content || '').slice(0, 250))}..."</p>
                  </div>
                `).join('');
              } else if (contextPanel) {
                contextPanel.innerHTML = '<div class="context-empty"><p>No context chunks retrieved</p></div>';
              }
              messages.scrollTop = messages.scrollHeight;
            }

            if (event.type === 'error') {
              throw new Error(event.error);
            }
          } catch (parseErr) {
            // Ignore parse errors for partial/malformed JSON, but log them for debugging
            if (parseErr.message !== 'Unexpected end of JSON input') {
              console.warn('SSE parse error:', parseErr.message, 'Payload:', payload);
            }
          }
        }
      }
    }
  } catch (err) {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();

    const errorMsg = document.createElement('div');
    errorMsg.className = 'chat-message assistant';
    errorMsg.innerHTML = `
      <div class="message-bubble" style="border-color: rgba(255, 85, 85, 0.2);">
        <span style="color: #ff5555; font-size: 0.82rem;">⚠ ${escapeHtml(err.message)}</span>
        <br><em style="color: var(--text-muted); font-size: 0.78rem;">Make sure the FastAPI backend is running and your .env has valid API keys.</em>
      </div>
      <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    messages.appendChild(errorMsg);
    messages.scrollTop = messages.scrollHeight;
  } finally {
    input.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ── Enter key to send ──
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && document.activeElement?.id === 'chat-input') {
    sendMessage();
  }
});

// ── Feature card hover glow tracking ──
function initCardGlow() {
  const cards = document.querySelectorAll('.feature-card');
  cards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const glow = card.querySelector('.card-glow');
      if (glow) {
        glow.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(88,166,255,0.06), transparent 60%)`;
      }
    });
  });
}

// ── Initialize on page load ──
document.addEventListener('DOMContentLoaded', () => {
  // Check initial route
  if (window.location.pathname === '/analyst') {
    navigateToAnalyst();
  }

  initChartAnimation();
  initScrollReveal();
  initCardStagger();
  initArchStagger();
  initCardGlow();
});

// ── Document Modal logic ──
async function openDocument(filename, title) {
  const modal = document.getElementById('doc-modal');
  const titleEl = document.getElementById('doc-modal-title');
  const bodyEl = document.getElementById('doc-modal-body');

  if (!modal || !titleEl || !bodyEl) return;

  titleEl.textContent = title;
  bodyEl.innerHTML = '<p>Loading document text...</p>';
  modal.setAttribute('aria-hidden', 'false');

  try {
    const response = await fetch('/raw/' + filename);
    if (!response.ok) throw new Error('Document not found');

    const text = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, 'text/html');

    // Cleanly extract text from paragraphs
    const paragraphs = doc.querySelectorAll('p');
    if (paragraphs.length > 0) {
      bodyEl.innerHTML = '';
      paragraphs.forEach(p => {
        const textContent = p.textContent.trim();
        // Skip overly short / empty artifacts from raw HTML parsing
        if (textContent && textContent.length > 5) {
          const pEl = document.createElement('p');
          pEl.textContent = textContent;
          bodyEl.appendChild(pEl);
        }
      });
    } else {
      const pEl = document.createElement('p');
      pEl.textContent = doc.body.textContent.trim();
      bodyEl.appendChild(pEl);
    }
  } catch (err) {
    bodyEl.innerHTML = `<p style="color:#ff5555">Error loading document: ${escapeHtml(err.message)}</p>`;
  }
}

function closeDocument() {
  const modal = document.getElementById('doc-modal');
  if (modal) {
    modal.setAttribute('aria-hidden', 'true');
  }
}
