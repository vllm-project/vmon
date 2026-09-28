// SPDX-License-Identifier: Apache-2.0

function makeChart(id, title, datasets) {
  const container = document.getElementById(id);
  const heading = document.createElement('h2');
  heading.textContent = title;
  container.append(heading);

  const legend = document.createElement('div');
  legend.className = 'legend';
  container.append(legend);

  const svgNS = 'http://www.w3.org/2000/svg';
  function element(tag, attrs, text) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const svg = element('svg', {
    viewBox: '0 0 720 280', role: 'img', tabindex: '0',
    'aria-label': title + '. Use the arrow keys to inspect samples.',
  });
  container.append(svg);
  const tooltip = document.createElement('pre');
  tooltip.className = 'chart-values';
  tooltip.textContent = 'Hover over the chart or use the arrow keys to inspect samples.';
  container.append(tooltip);

  const hidden = new Set();
  const times = DATA.map((sample, i) => Number.isFinite(sample.elapsed_secs) ? sample.elapsed_secs : i);
  const left = 72, top = 12, bottom = 242;
  let width = 720, right = width - 28;
  let timeMin = 0, timeMax = 1;
  if (times.length) {
    timeMin = times[0];
    timeMax = Math.max(timeMin + 1, times[times.length - 1]);
  }
  const x = time => left + (time - timeMin) / (timeMax - timeMin) * (right - left);
  const formatter = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });
  const format = value => Number.isFinite(value) ? formatter.format(value) : '—';
  let cursor;
  let selected = -1;

  function inspect(index) {
    if (!times.length) return;
    selected = Math.max(0, Math.min(times.length - 1, index));
    const lines = [format(times[selected]) + 's'];
    datasets.forEach((dataset, i) => {
      if (!hidden.has(i)) lines.push(dataset.label + ': ' + format(dataset.data[selected]));
    });
    tooltip.textContent = lines.join('\n');
    cursor.setAttribute('x1', x(times[selected]));
    cursor.setAttribute('x2', x(times[selected]));
    cursor.setAttribute('visibility', 'visible');
  }

  function draw() {
    svg.replaceChildren();
    let low = 0, high = 0, count = 0;
    datasets.forEach((dataset, i) => {
      if (hidden.has(i)) return;
      for (const value of dataset.data) {
        if (!Number.isFinite(value)) continue;
        low = Math.min(low, value);
        high = Math.max(high, value);
        count++;
      }
    });
    if (high === low) high = low + 1;
    const y = value => bottom - (value - low) / (high - low) * (bottom - top);
    for (let i = 0; i <= 4; i++) {
      const value = low + (high - low) * i / 4;
      svg.append(element('line', {x1: left, x2: right, y1: y(value), y2: y(value), stroke: '#334155'}));
      svg.append(element('text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end'}, format(value)));
      const time = timeMin + (timeMax - timeMin) * i / 4;
      svg.append(element('text', {x: x(time), y: bottom + 24, 'text-anchor': 'middle'}, format(time) + 's'));
    }
    datasets.forEach((dataset, i) => {
      if (hidden.has(i)) return;
      const points = [];
      dataset.data.forEach((value, j) => {
        if (Number.isFinite(value) && j < times.length) points.push([x(times[j]), y(value)]);
      });
      if (points.length === 1) {
        svg.append(element('circle', {cx: points[0][0], cy: points[0][1], r: 3, fill: dataset.color}));
      } else if (points.length > 1) {
        svg.append(element('path', {
          d: points.map(([px, py], j) => (j ? 'L' : 'M') + px.toFixed(2) + ',' + py.toFixed(2)).join(' '),
          fill: 'none', stroke: dataset.color, 'stroke-width': 1.5,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
    });
    if (!count) {
      svg.append(element('text', {x: (left + right) / 2, y: 120, 'text-anchor': 'middle'},
        times.length ? 'No data for visible series' : 'No samples'));
    }
    cursor = element('line', {
      x1: left, x2: left, y1: top, y2: bottom, stroke: '#e2e8f0',
      'stroke-dasharray': '4 4', visibility: 'hidden',
    });
    svg.append(cursor);
    if (selected >= 0) inspect(selected);
  }

  datasets.forEach((dataset, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = dataset.label;
    button.style.color = dataset.color;
    button.setAttribute('aria-pressed', 'true');
    button.addEventListener('click', () => {
      if (hidden.has(i)) hidden.delete(i); else hidden.add(i);
      button.setAttribute('aria-pressed', String(!hidden.has(i)));
      draw();
    });
    legend.append(button);
  });

  svg.addEventListener('pointermove', event => {
    if (!times.length) return;
    // Convert screen coordinates through the SVG viewBox transform.
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const position = point.matrixTransform(svg.getScreenCTM().inverse());
    const time = timeMin + Math.max(0, Math.min(1, (position.x - left) / (right - left))) * (timeMax - timeMin);
    let lo = 0, hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (times[mid] < time) lo = mid + 1; else hi = mid;
    }
    inspect(lo > 0 && time - times[lo - 1] < times[lo] - time ? lo - 1 : lo);
  });
  svg.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    inspect(selected < 0 ? 0 : selected + (event.key === 'ArrowRight' ? 1 : -1));
  });
  new ResizeObserver(() => {
    const nextWidth = Math.max(240, svg.clientWidth);
    if (nextWidth === width) return;
    width = nextWidth;
    right = width - 28;
    svg.setAttribute('viewBox', '0 0 ' + width + ' 280');
    draw();
  }).observe(svg);
  draw();
}
