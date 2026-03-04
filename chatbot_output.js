(function () {
    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function formatResponse(text) {
        if (typeof text !== 'string') {
            text = JSON.stringify(text, null, 2);
        }

        const trimmed = text.trim();
        const fencedMatch = trimmed.match(/^```(\w+)?\r?\n([\s\S]*?)```$/);
        if (fencedMatch) {
            const lang = (fencedMatch[1] || 'text').toLowerCase();
            const code = fencedMatch[2] || '';
            return `<pre><code class="lang-${lang}">${escapeHtml(code)}</code></pre>`;
        }

        const looksLikeJson = (
            (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
            (trimmed.startsWith('[') && trimmed.endsWith(']'))
        );
        if (looksLikeJson) {
            return `<pre><code class="lang-json">${escapeHtml(trimmed)}</code></pre>`;
        }

        const codeBlocks = [];
        text = text.replace(/```(\w+)?\r?\n([\s\S]*?)```/g, (_, lang, code) => {
            const token = `__CODE_BLOCK_${codeBlocks.length}__`;
            codeBlocks.push({
                token,
                html: `<pre><code class="lang-${(lang || 'text').toLowerCase()}">${escapeHtml(code)}</code></pre>`,
            });
            return token;
        });

        text = escapeHtml(text);
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        text = text.replace(/\n/g, '<br>');
        text = text.replace(/^(\s*)-\s+(.+)/gm, '<li style="margin-left:20px;">$2</li>');
        text = text.replace(/^(\s*)\d+\.\s+(.+)/gm, '<li style="margin-left:20px;">$2</li>');

        for (const block of codeBlocks) {
            text = text.replace(escapeHtml(block.token), block.html);
        }

        return text;
    }

    function detectOutputCommand(query) {
        const text = (query || '').toLowerCase();
        const wantsDownload = /\b(download|export|save)\b/.test(text);
        const wantsRender = /\b(show|display|format|convert)\b/.test(text);
        const wantsVisual = /\b(visual|visualize|visualise|visualization|visualisation|viusal|visul|chart|graph|plot|representation)\b/.test(text);
        const wantsTable = /\b(table|tabular)\b/.test(text);
        const chartType = /\bpie\b/.test(text)
            ? 'pie'
            : /\bline\b/.test(text)
                ? 'line'
                : /\bbar\b/.test(text)
                    ? 'bar'
                    : 'bar';
        const startsWithCommand = /^(show|display|format|convert|download|export|save|give|tell|provide)\b/.test(text.trim());
        const mentionsPriorResult = /\b(output|result|response|data|info|it|this|last|their|them|those)\b/.test(text);
        const mentionsFile = /\b(file|document|attachment)\b/.test(text);
        const conversationalDownload = /\blet me\b/.test(text) && wantsDownload;
        const referencesExisting = startsWithCommand || mentionsPriorResult || mentionsFile || conversationalDownload;
        const likelyDomainQuery = /\b(personnel|district|unit|transfer|vacanc|village|si|constable|inspector|rank)\b/.test(text);
        const format =
            /\bjson\b/.test(text) ? 'json' :
            /\b(tree|hierarchy)\b/.test(text) ? 'tree' :
            /\b(text|txt|plain)\b/.test(text) ? 'text' :
            null;

        if ((wantsDownload || wantsRender) && format && referencesExisting && !likelyDomainQuery) {
            return { kind: wantsDownload ? 'download' : 'render', format };
        }
        if (wantsDownload && !format && referencesExisting && !likelyDomainQuery) {
            return { kind: 'download', format: 'text' };
        }
        if (wantsTable && referencesExisting && !wantsDownload) {
            return { kind: 'table', format: 'table' };
        }
        if (wantsVisual || /\b(pie chart|bar chart|line chart)\b/.test(text)) {
            return { kind: 'visual', format: chartType };
        }
        return null;
    }

    function toTreeText(value, name = 'result', prefix = '', isLast = true, depth = 0) {
        const maxDepth = 6;
        const maxItems = 20;
        const connector = isLast ? '`- ' : '|- ';
        const lines = [`${prefix}${connector}${name}`];
        if (depth >= maxDepth) {
            lines.push(`${prefix}${isLast ? '   ' : '|  '}\`- ...`);
            return lines.join('\n');
        }

        const childPrefix = prefix + (isLast ? '   ' : '|  ');
        if (Array.isArray(value)) {
            if (value.length === 0) return `${prefix}${connector}${name}: []`;
            const slice = value.slice(0, maxItems);
            slice.forEach((item, idx) => {
                const childLast = idx === slice.length - 1 && value.length <= maxItems;
                if (item && typeof item === 'object') {
                    lines.push(toTreeText(item, `[${idx}]`, childPrefix, childLast, depth + 1));
                } else {
                    lines.push(`${childPrefix}${childLast ? '`- ' : '|- '}[${idx}]: ${JSON.stringify(item)}`);
                }
            });
            if (value.length > maxItems) {
                lines.push(`${childPrefix}\`- ... (${value.length - maxItems} more items)`);
            }
            return lines.join('\n');
        }

        if (value && typeof value === 'object') {
            const entries = Object.entries(value);
            if (entries.length === 0) return `${prefix}${connector}${name}: {}`;
            const slice = entries.slice(0, maxItems);
            slice.forEach(([k, v], idx) => {
                const childLast = idx === slice.length - 1 && entries.length <= maxItems;
                if (v && typeof v === 'object') {
                    lines.push(toTreeText(v, k, childPrefix, childLast, depth + 1));
                } else {
                    lines.push(`${childPrefix}${childLast ? '`- ' : '|- '}${k}: ${JSON.stringify(v)}`);
                }
            });
            if (entries.length > maxItems) {
                lines.push(`${childPrefix}\`- ... (${entries.length - maxItems} more keys)`);
            }
            return lines.join('\n');
        }

        return `${prefix}${connector}${name}: ${JSON.stringify(value)}`;
    }

    function renderFromLast(lastAssistantResult, format) {
        if (!lastAssistantResult) return null;
        const toolResult = (lastAssistantResult.data && typeof lastAssistantResult.data === 'object')
            ? lastAssistantResult.data
            : null;
        const base = (toolResult && toolResult.data !== undefined)
            ? toolResult.data
            : (toolResult || lastAssistantResult);
        if (!base) return null;
        if (format === 'json') return JSON.stringify(base, null, 2);
        if (format === 'tree') return toTreeText(base, 'data');
        return lastAssistantResult.response || JSON.stringify(base, null, 2);
    }

    function getLastBase(lastAssistantResult) {
        if (!lastAssistantResult) return null;
        const toolResult = (lastAssistantResult.data && typeof lastAssistantResult.data === 'object')
            ? lastAssistantResult.data
            : null;
        return (toolResult && toolResult.data !== undefined)
            ? toolResult.data
            : (toolResult || lastAssistantResult);
    }

    function extractChartSeries(base) {
        if (!base) return null;

        if (base && typeof base === 'object' && Array.isArray(base.distribution)) {
            const series = base.distribution
                .map((item) => ({
                    label: item.rankName || item.districtName || item.unitTypeName || item.name || 'Unknown',
                    value: Number(item.count || 0),
                }))
                .filter((item) => Number.isFinite(item.value) && item.value >= 0);
            return series.length ? series : null;
        }

        if (Array.isArray(base)) {
            const numericList = base
                .map((item) => ({
                    label: item?.rankName || item?.districtName || item?.unitTypeName || item?.name || item?.label || 'Unknown',
                    value: Number(item?.count ?? item?.value ?? item?.personnelCount ?? item?.totalPersonnel ?? item?.villageCount ?? 0),
                }))
                .filter((item) => Number.isFinite(item.value) && item.value >= 0);
            return numericList.length ? numericList : null;
        }

        if (base && typeof base === 'object') {
            const entries = Object.entries(base)
                .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
                .map(([label, value]) => ({ label, value }));
            return entries.length ? entries : null;
        }

        return null;
    }

    function renderVisualizationFromLast(lastAssistantResult, chartType = 'bar') {
        const base = getLastBase(lastAssistantResult);
        const series = extractChartSeries(base);
        if (!series) return null;

        const sorted = [...series].sort((a, b) => b.value - a.value).slice(0, 12);
        const maxValue = Math.max(...sorted.map((item) => item.value), 1);
        const total = sorted.reduce((acc, item) => acc + item.value, 0);

        if (chartType === 'pie') {
            const palette = ['#2d6aa4', '#1c9fd6', '#1d4e89', '#0b7285', '#2b8a3e', '#c08000', '#b02a37', '#6f42c1', '#7d8597', '#495057', '#9c36b5', '#0ca678'];
            let cursor = 0;
            const slices = sorted.map((item, idx) => {
                const pct = total > 0 ? (item.value / total) * 100 : 0;
                const start = cursor;
                const end = cursor + pct;
                cursor = end;
                return { ...item, pct, color: palette[idx % palette.length], start, end };
            });
            const gradient = slices.map((s) => `${s.color} ${s.start}% ${s.end}%`).join(', ');
            const legend = slices.map((s) => `
                <div class="viz-legend-item">
                    <span class="viz-dot" style="background:${s.color}"></span>
                    <span class="viz-legend-label">${escapeHtml(s.label)}</span>
                    <span class="viz-legend-value">${s.value} (${s.pct.toFixed(1)}%)</span>
                </div>
            `).join('');
            return `
                <div class="viz-card">
                    <div class="viz-title">Data Visualization</div>
                    <div class="viz-subtitle">Pie chart &bull; ${sorted.length} categories${total ? ` &bull; Total ${total}` : ''}</div>
                    <div class="viz-pie-wrap">
                        <div class="viz-pie" style="background: conic-gradient(${gradient})"></div>
                    </div>
                    <div class="viz-legend">${legend}</div>
                </div>
            `;
        }

        const rows = sorted.map((item) => {
            const pct = Math.max(4, Math.round((item.value / maxValue) * 100));
            return `
                <div class="viz-row">
                    <div class="viz-row-head">
                        <span class="viz-label">${escapeHtml(item.label)}</span>
                        <span class="viz-value">${item.value}</span>
                    </div>
                    <div class="viz-bar-wrap">
                        <div class="viz-bar" style="width:${pct}%"></div>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="viz-card">
                <div class="viz-title">Data Visualization</div>
                <div class="viz-subtitle">Bar chart &bull; Top ${sorted.length} categories${total ? ` &bull; Total ${total}` : ''}</div>
                ${rows}
            </div>
        `;
    }

    function renderTableFromLast(lastAssistantResult) {
        const base = getLastBase(lastAssistantResult);
        if (!base) return null;

        const rows = Array.isArray(base) ? base : [base];
        const objectRows = rows.filter((r) => r && typeof r === 'object' && !Array.isArray(r));
        if (!objectRows.length) return null;

        const keys = [];
        for (const row of objectRows.slice(0, 50)) {
            for (const key of Object.keys(row)) {
                if (!keys.includes(key) && keys.length < 12) keys.push(key);
            }
            if (keys.length >= 12) break;
        }
        if (!keys.length) return null;

        const header = keys.map((k) => `<th>${escapeHtml(k)}</th>`).join('');
        const body = objectRows.slice(0, 30).map((row) => {
            const cells = keys.map((k) => {
                const value = row[k];
                const display = (value && typeof value === 'object')
                    ? JSON.stringify(value)
                    : String(value ?? '');
                return `<td>${escapeHtml(display)}</td>`;
            }).join('');
            return `<tr>${cells}</tr>`;
        }).join('');

        return `
            <div class="table-card">
                <div class="table-title">Data Table</div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr>${header}</tr></thead>
                        <tbody>${body}</tbody>
                    </table>
                </div>
            </div>
        `;
    }

    function triggerDownload(payload) {
        if (!payload || !payload.download || !payload.download.content) return;
        const content = payload.download.content;
        const filename = payload.download.filename || `query_result.${payload.format === 'json' ? 'json' : 'txt'}`;
        const contentType = payload.download.content_type || 'text/plain';
        const blob = new Blob([content], { type: contentType });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(link.href);
    }

    window.OutputLayer = {
        formatResponse,
        detectOutputCommand,
        renderFromLast,
        renderVisualizationFromLast,
        renderTableFromLast,
        triggerDownload,
    };
})();

