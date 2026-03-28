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

// ── Chat Interface (Demo) ──
function sendMessage() {
  const input = document.getElementById('chat-input');
  const messages = document.getElementById('chat-messages');
  const contextPanel = document.getElementById('context-panel');

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
  messages.scrollTop = messages.scrollHeight;

  // Simulate assistant response
  setTimeout(() => {
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'chat-message assistant';
    assistantMsg.innerHTML = `
      <div class="message-bubble">
        <em style="color: var(--text-muted); font-size: 0.82rem;">
          This is a demo interface. Connect to the FastAPI backend at <code style="color: var(--accent); font-size: 0.78rem;">/api/query</code> to get real FOMC analysis powered by LangGraph.
        </em>
      </div>
      <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    messages.appendChild(assistantMsg);
    messages.scrollTop = messages.scrollHeight;

    // Add sample context chunks
    if (contextPanel) {
      contextPanel.innerHTML = `
        <div class="context-chunk">
          <div class="chunk-source">
            <span class="chunk-badge">STMT</span>
            <span class="chunk-label">Sep 2024 Statement</span>
          </div>
          <p class="chunk-text">"The Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent..."</p>
        </div>
        <div class="context-chunk">
          <div class="chunk-source">
            <span class="chunk-badge">MIN</span>
            <span class="chunk-label">Sep 2024 Minutes</span>
          </div>
          <p class="chunk-text">"A substantial majority of participants supported lowering the target range for the federal funds rate by 50 basis points..."</p>
        </div>
        <div class="context-chunk">
          <div class="chunk-source">
            <span class="chunk-badge">STMT</span>
            <span class="chunk-label">Jan 2025 Statement</span>
          </div>
          <p class="chunk-text">"The Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent..."</p>
        </div>
      `;
    }
  }, 800);
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
