/* ============================================
   TUSK AI - Shared JavaScript
   ============================================ */

// ---- Navbar Scroll Effect ----
const navbar = document.querySelector('.navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });
}

// ---- Mobile Menu ----
const hamburger = document.querySelector('.nav-hamburger');
const mobileMenu = document.querySelector('.mobile-menu');
if (hamburger && mobileMenu) {
  hamburger.addEventListener('click', () => {
    mobileMenu.classList.toggle('open');
    const spans = hamburger.querySelectorAll('span');
    const isOpen = mobileMenu.classList.contains('open');
    if (isOpen) {
      spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
      spans[1].style.opacity = '0';
      spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
    } else {
      spans.forEach(s => { s.style.transform = ''; s.style.opacity = ''; });
    }
  });
  // Close on link click
  mobileMenu.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => mobileMenu.classList.remove('open'));
  });
}

// ---- Animated Counter ----
function animateCounter(el, end, duration = 2000, suffix = '') {
  const start = 0;
  const increment = end / (duration / 16);
  let current = start;
  const isDecimal = String(end).includes('.');
  const timer = setInterval(() => {
    current += increment;
    if (current >= end) {
      current = end;
      clearInterval(timer);
    }
    el.textContent = (isDecimal ? current.toFixed(1) : Math.floor(current).toLocaleString()) + suffix;
  }, 16);
}

// ---- Intersection Observer for Counters & Animations ----
const observerOptions = { threshold: 0.3 };
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      // Trigger counter
      if (el.dataset.count !== undefined) {
        const end = parseFloat(el.dataset.count);
        const suffix = el.dataset.suffix || '';
        animateCounter(el, end, 1800, suffix);
        observer.unobserve(el);
      }
      // Trigger fade-in
      if (el.classList.contains('observe-fade')) {
        el.classList.add('fade-in');
        observer.unobserve(el);
      }
    }
  });
}, observerOptions);

document.querySelectorAll('[data-count]').forEach(el => observer.observe(el));
document.querySelectorAll('.observe-fade').forEach(el => observer.observe(el));

// ---- Set Active Nav Link ----
(function setActiveNav() {
  const current = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a, .mobile-menu a').forEach(a => {
    if (a.getAttribute('href') === current || (current === '' && a.getAttribute('href') === 'index.html')) {
      a.classList.add('active');
    }
  });
})();

// ---- Smooth reveal on scroll ----
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.reveal').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
  revealObserver.observe(el);
});

// ---- Ticker / Live counter ----
function startLiveTicker(selector, base, fluctuation = 3, interval = 2500) {
  const el = document.querySelector(selector);
  if (!el) return;
  let val = base;
  setInterval(() => {
    val += Math.floor(Math.random() * fluctuation);
    el.textContent = val.toLocaleString();
  }, interval);
}

startLiveTicker('#ticker-transactions', 1464, 5, 3000);
startLiveTicker('#ticker-threats', 9854, 2, 4000);

console.log('%cTUSK AI Platform', 'color:#10B981;font-size:1.5rem;font-weight:800;');
console.log('%cInitialized successfully.', 'color:#64748B');
