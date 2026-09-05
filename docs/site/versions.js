(() => {
  const root = new URL('../../', document.currentScript.src);
  const active = new URL('../', document.currentScript.src).pathname.split('/').filter(Boolean).pop();
  fetch(new URL('versions.json', root))
    .then(response => { if (!response.ok) throw new Error('Version index unavailable'); return response.json(); })
    .then(versions => {
      const select = document.createElement('select');
      select.setAttribute('aria-label', 'Documentation version');
      select.className = 'form-select form-select-sm w-auto';
      for (const version of versions) {
        if (!/^(candidate|[0-9][a-zA-Z0-9.+-]*)$/.test(version)) continue;
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version === 'candidate' ? 'Candidate (unpublished)' : version;
        option.selected = version === active;
        select.append(option);
      }
      select.addEventListener('change', () => { window.location.href = new URL(`${select.value}/`, root).href; });
      document.querySelector('.navbar-brand')?.after(select);
    })
    .catch(() => { /* The version remains visible in the page title when offline. */ });
})();
