(function () {
    const defaultApiUrl = `${window.location.protocol}//${window.location.hostname}:8090`;
    const config = {
        apiUrl: defaultApiUrl,
    };
    const DEMO_USERNAME = '9999';
    const DEMO_PIN = '9999';
    const AUTH_STORAGE_KEY = 'mcpDemoAuth';
    const AUTH_BEARER_STORAGE_KEY = 'mcpAuthBearerToken';
    const AUTH_LOCKOUT_KEY = 'mcpDemoAuthLockoutUntil';
    const MAX_LOGIN_ATTEMPTS = 5;
    const LOCKOUT_MS = 60 * 1000;
    let lastAssistantResult = null;
    const responseBubblePagers = new Map();
    let responseBubblePagerCounter = 0;
    const backendContextDesyncWarnedChats = new Set();
    let loginFailedAttempts = 0;
    let lockoutIntervalRef = null;
    const CHAT_CLIENT_ID_STORAGE_KEY = 'mcpChatClientId';
    let chatThreadsState = null;
    let chatThreadsReadyPromise = null;

    function generateId(prefix = 'id') {
        if (window.crypto && window.crypto.randomUUID) {
            return `${prefix}-${window.crypto.randomUUID()}`;
        }
        return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function generateSessionId() {
        return generateId('session');
    }

    function getOrCreateChatClientId() {
        try {
            const existing = localStorage.getItem(CHAT_CLIENT_ID_STORAGE_KEY);
            if (existing) return existing;
            const created = generateId('client');
            localStorage.setItem(CHAT_CLIENT_ID_STORAGE_KEY, created);
            return created;
        } catch (_error) {
            return generateId('client');
        }
    }

    function getStoredBearerToken() {
        const candidates = [];
        try { candidates.push(localStorage.getItem(AUTH_BEARER_STORAGE_KEY)); } catch (_e) {}
        try { candidates.push(sessionStorage.getItem(AUTH_BEARER_STORAGE_KEY)); } catch (_e) {}
        try { candidates.push(window.MCP_AUTH_TOKEN); } catch (_e) {}
        try { candidates.push(window.__MCP_AUTH_TOKEN__); } catch (_e) {}

        for (const rawValue of candidates) {
            const value = typeof rawValue === 'string' ? rawValue.trim() : '';
            if (!value) continue;
            return value.toLowerCase().startsWith('bearer ') ? value : `Bearer ${value}`;
        }
        return null;
    }

    function buildApiHeaders(extraHeaders = {}) {
        const headers = {
            'Content-Type': 'application/json',
            'X-Chat-Client-ID': getOrCreateChatClientId(),
        };
        const bearerToken = getStoredBearerToken();
        if (bearerToken) {
            headers.Authorization = bearerToken;
        }
        return {
            ...headers,
            ...extraHeaders,
        };
    }

    async function chatApiFetch(path, options = {}) {
        const headers = buildApiHeaders(options.headers || {});
        const response = await fetch(`${config.apiUrl}${path}`, {
            ...options,
            headers,
        });
        if (!response.ok) {
            let message = `HTTP ${response.status}`;
            try {
                const payload = await response.json();
                message = payload?.detail?.message || payload?.detail || payload?.error?.message || message;
            } catch (_ignored) {
                // Keep generic message.
            }
            const error = new Error(message);
            error.status = response.status;
            throw error;
        }
        return response.json();
    }

    function nowIso() {
        return new Date().toISOString();
    }

    function compactWhitespace(value) {
        return String(value || '')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function truncateText(value, maxLength = 56) {
        const text = compactWhitespace(value);
        if (text.length <= maxLength) return text;
        return `${text.slice(0, maxLength - 1).trimEnd()}...`;
    }

    function formatChatTitleDate(dateLike) {
        const date = new Date(dateLike || Date.now());
        if (Number.isNaN(date.getTime())) return 'Unknown date';
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    }

    function formatChatListMeta(dateLike) {
        const date = new Date(dateLike || Date.now());
        if (Number.isNaN(date.getTime())) return '';
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    function buildChatTitle(chat) {
        const messages = Array.isArray(chat?.messages) ? chat.messages : [];
        const userQuestions = messages
            .filter((message) => message && message.role === 'user' && compactWhitespace(message.content))
            .slice(0, 2)
            .map((message) => truncateText(message.content, 42));
        const datePart = formatChatTitleDate(chat?.createdAt);

        if (userQuestions.length === 0) {
            return `New Chat - ${datePart}`;
        }

        const questionPart = userQuestions.join(' / ');
        return `${questionPart} - ${datePart}`;
    }

    function createChatThread(overrides = {}) {
        const createdAt = overrides.createdAt || nowIso();
        const thread = {
            id: overrides.id || generateId('chat'),
            sessionId: overrides.sessionId || generateSessionId(),
            createdAt,
            updatedAt: overrides.updatedAt || createdAt,
            messages: Array.isArray(overrides.messages) ? overrides.messages : [],
            lastAssistantResult: overrides.lastAssistantResult ?? null,
            title: overrides.title || '',
        };
        thread.title = buildChatTitle(thread);
        return thread;
    }

    function normalizeMessageRecord(message) {
        if (!message || typeof message !== 'object') return null;
        const role = message.role === 'user' ? 'user' : 'assistant';
        const content = typeof message.content === 'string' ? message.content : String(message.content ?? '');
        const timestamp = typeof message.timestamp === 'string' ? message.timestamp : nowIso();
        return {
            role,
            content,
            rawHtml: Boolean(message.rawHtml),
            tone: message.tone === 'error' ? 'error' : 'normal',
            timestamp,
        };
    }

    function normalizeChatThread(thread) {
        if (!thread || typeof thread !== 'object') return null;
        const normalizedMessages = Array.isArray(thread.messages)
            ? thread.messages.map(normalizeMessageRecord).filter(Boolean)
            : [];
        const normalized = createChatThread({
            id: typeof thread.id === 'string' && thread.id ? thread.id : undefined,
            sessionId: typeof thread.sessionId === 'string' && thread.sessionId ? thread.sessionId : undefined,
            createdAt: typeof thread.createdAt === 'string' ? thread.createdAt : undefined,
            updatedAt: typeof thread.updatedAt === 'string' ? thread.updatedAt : undefined,
            messages: normalizedMessages,
            lastAssistantResult: thread.lastAssistantResult ?? null,
            title: typeof thread.title === 'string' ? thread.title : undefined,
        });
        normalized.updatedAt = normalizedMessages.length
            ? normalizedMessages[normalizedMessages.length - 1].timestamp
            : normalized.updatedAt;
        normalized.title = buildChatTitle(normalized);
        return normalized;
    }

    function buildDefaultChatThreadsState() {
        const firstChat = createChatThread();
        return {
            activeChatId: firstChat.id,
            chats: [firstChat],
        };
    }

    function saveChatThreadsState() {
        // Chat memory now lives on the server. Keep only in-memory UI state here.
        return;
    }

    async function loadChatThreadsState() {
        try {
            const payload = await chatApiFetch('/api/v1/chats', { method: 'GET' });
            let chats = Array.isArray(payload?.chats)
                ? payload.chats.map(normalizeChatThread).filter(Boolean)
                : [];
            if (!chats.length) {
                const created = await chatApiFetch('/api/v1/chats', {
                    method: 'POST',
                    body: JSON.stringify({}),
                });
                const chat = normalizeChatThread(created?.chat);
                chats = chat ? [chat] : [];
            }
            if (!chats.length) {
                chatThreadsState = buildDefaultChatThreadsState();
                return;
            }

            const activeChatId = typeof payload?.active_chat_id === 'string'
                ? payload.active_chat_id
                : chats[0].id;
            chatThreadsState = {
                activeChatId: chats.some((chat) => chat.id === activeChatId) ? activeChatId : chats[0].id,
                chats: chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)),
            };
        } catch (error) {
            console.warn('Failed to load chat threads from server, using local in-memory fallback', error);
            chatThreadsState = buildDefaultChatThreadsState();
        }
    }

    async function ensureChatThreadsStateReady() {
        if (chatThreadsState?.chats?.length) {
            return chatThreadsState;
        }
        if (!chatThreadsReadyPromise) {
            chatThreadsReadyPromise = (async () => {
                await loadChatThreadsState();
                if (!chatThreadsState?.chats?.length) {
                    chatThreadsState = buildDefaultChatThreadsState();
                }
                return chatThreadsState;
            })().finally(() => {
                chatThreadsReadyPromise = null;
            });
        }
        return await chatThreadsReadyPromise;
    }

    function ensureChatThreadsState() {
        // Async loading is explicit via ensureChatThreadsStateReady().
        if (!chatThreadsState?.chats?.length) {
            chatThreadsState = buildDefaultChatThreadsState();
            saveChatThreadsState();
        }
        return chatThreadsState;
    }

    function getActiveChat() {
        const state = ensureChatThreadsState();
        return state.chats.find((chat) => chat.id === state.activeChatId) || state.chats[0] || null;
    }

    function touchActiveChatMeta() {
        const activeChat = getActiveChat();
        if (!activeChat) return;
        activeChat.updatedAt = activeChat.messages.length
            ? activeChat.messages[activeChat.messages.length - 1].timestamp
            : activeChat.updatedAt || activeChat.createdAt || nowIso();
        activeChat.title = buildChatTitle(activeChat);
        chatThreadsState.chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        chatThreadsState.activeChatId = activeChat.id;
        saveChatThreadsState();
    }

    function getActiveChatSessionId() {
        const activeChat = getActiveChat();
        return activeChat?.sessionId || generateSessionId();
    }

    function setActiveChatLastAssistantResult(result) {
        const activeChat = getActiveChat();
        if (!activeChat) return;
        activeChat.lastAssistantResult = result ?? null;
        touchActiveChatMeta();
        saveChatThreadsState();
    }

    async function createNewChat() {
        await ensureChatThreadsStateReady();
        try {
            const created = await chatApiFetch('/api/v1/chats', {
                method: 'POST',
                body: JSON.stringify({}),
            });
            const newChat = normalizeChatThread(created?.chat) || createChatThread();
            chatThreadsState.chats.unshift(newChat);
            chatThreadsState.chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
            chatThreadsState.activeChatId = newChat.id;
            return newChat;
        } catch (error) {
            console.warn('Failed to create server chat; using local fallback', error);
            const newChat = createChatThread();
            chatThreadsState.chats.unshift(newChat);
            chatThreadsState.activeChatId = newChat.id;
            return newChat;
        }
    }

    async function deleteChat(chatId) {
        const state = await ensureChatThreadsStateReady();
        try {
            const payload = await chatApiFetch(`/api/v1/chats/${encodeURIComponent(chatId)}`, {
                method: 'DELETE',
            });
            const chats = Array.isArray(payload?.chats)
                ? payload.chats.map(normalizeChatThread).filter(Boolean)
                : [];
            if (chats.length) {
                state.chats = chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
                state.activeChatId = (typeof payload?.active_chat_id === 'string' && payload.active_chat_id)
                    ? payload.active_chat_id
                    : state.chats[0].id;
                const active = state.chats.find((chat) => chat.id === state.activeChatId) || state.chats[0];
                lastAssistantResult = active?.lastAssistantResult ?? null;
            } else {
                const replacement = createChatThread();
                state.chats = [replacement];
                state.activeChatId = replacement.id;
                lastAssistantResult = null;
            }
        } catch (error) {
            console.warn('Failed to delete chat on server; applying local delete only', error);
            const targetIndex = state.chats.findIndex((chat) => chat.id === chatId);
            if (targetIndex < 0) return;
            state.chats.splice(targetIndex, 1);
            if (!state.chats.length) {
                const replacement = createChatThread();
                state.chats = [replacement];
                state.activeChatId = replacement.id;
                lastAssistantResult = null;
            } else if (state.activeChatId === chatId) {
                state.activeChatId = state.chats[0].id;
                lastAssistantResult = state.chats[0].lastAssistantResult ?? null;
            }
        }
        renderActiveChatConversation();
        setConnectionStatus(true, 'Ready');
    }

    async function clearAllChats() {
        await ensureChatThreadsStateReady();
        if (!window.confirm('Clear all chat threads from the server-side session? This cannot be undone.')) {
            return;
        }

        try {
            const payload = await chatApiFetch('/api/v1/chats', {
                method: 'DELETE',
                body: JSON.stringify({}),
            });
            const chats = Array.isArray(payload?.chats)
                ? payload.chats.map(normalizeChatThread).filter(Boolean)
                : [];
            const freshChat = chats[0] || createChatThread();
            chatThreadsState = {
                activeChatId: (typeof payload?.active_chat_id === 'string' && payload.active_chat_id) || freshChat.id,
                chats: chats.length ? chats : [freshChat],
            };
        } catch (error) {
            console.warn('Failed to clear chats on server; resetting local state only', error);
            const freshChat = createChatThread();
            chatThreadsState = {
                activeChatId: freshChat.id,
                chats: [freshChat],
            };
        }
        lastAssistantResult = null;
        backendContextDesyncWarnedChats.clear();
        renderActiveChatConversation();
        setConnectionStatus(true, 'Ready');
    }

    function renderChatThreadList() {
        const list = document.getElementById('chatThreadList');
        if (!list) return;

        const state = ensureChatThreadsState();
        list.innerHTML = '';

        state.chats.forEach((chat) => {
            const row = document.createElement('div');
            row.className = 'chat-thread-row';
            row.setAttribute('role', 'listitem');

            const button = document.createElement('button');
            button.type = 'button';
            button.className = `chat-thread-item${chat.id === state.activeChatId ? ' active' : ''}`;
            button.dataset.chatId = chat.id;
            button.innerHTML = `
                <span class="chat-thread-title">${chat.title}</span>
                <span class="chat-thread-meta">${formatChatListMeta(chat.updatedAt)}</span>
            `;
            button.addEventListener('click', () => {
                switchChat(chat.id);
            });

            const deleteBtn = document.createElement('button');
            deleteBtn.type = 'button';
            deleteBtn.className = 'chat-thread-delete-btn';
            deleteBtn.setAttribute('aria-label', `Delete chat: ${chat.title}`);
            deleteBtn.title = 'Delete chat';
            deleteBtn.innerHTML = `
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v8h-2V9zm4 0h2v8h-2V9zM7 9h2v8H7V9zm-1 12h12a2 2 0 0 0 2-2V8H4v11a2 2 0 0 0 2 2z"></path>
                </svg>
            `;
            deleteBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                const confirmed = window.confirm(`Delete this chat?\n\n${chat.title}`);
                if (!confirmed) return;
                void deleteChat(chat.id);
            });

            row.appendChild(button);
            row.appendChild(deleteBtn);
            list.appendChild(row);
        });
    }

    function updateActiveChatLabel() {
        const label = document.getElementById('activeChatLabel');
        if (!label) return;
        const activeChat = getActiveChat();
        label.textContent = activeChat
            ? `Current chat: ${activeChat.title}`
            : 'Current chat: New Chat';
    }

    function clearConversationUi() {
        removeTypingIndicator();
        responseBubblePagers.clear();
        responseBubblePagerCounter = 0;
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        messagesContainer.querySelectorAll('.message').forEach((messageNode) => messageNode.remove());
    }

    function showWelcome() {
        const welcome = document.getElementById('welcomeMessage');
        if (welcome) welcome.style.display = '';
    }

    function renderStoredMessage(messageRecord) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return null;
        const element = createMessage(
            messageRecord.content,
            messageRecord.role === 'user',
            Boolean(messageRecord.rawHtml),
            {
                timestamp: messageRecord.timestamp,
                tone: messageRecord.tone,
            }
        );
        messagesContainer.appendChild(element);
        return element;
    }

    function renderActiveChatConversation() {
        const activeChat = getActiveChat();
        clearConversationUi();
        if (!activeChat || !activeChat.messages.length) {
            showWelcome();
            lastAssistantResult = activeChat?.lastAssistantResult ?? null;
            updateActiveChatLabel();
            renderChatThreadList();
            return;
        }

        hideWelcome();
        activeChat.messages.forEach((messageRecord) => {
            const element = renderStoredMessage(messageRecord);
            if (
                element &&
                messageRecord.role !== 'user' &&
                !messageRecord.rawHtml &&
                messageRecord.tone !== 'error'
            ) {
                attachResponseBubblePager(messageRecord.content, element);
            }
        });
        lastAssistantResult = activeChat.lastAssistantResult ?? null;
        updateActiveChatLabel();
        renderChatThreadList();
        scrollToBottom();
    }

    function switchChat(chatId) {
        const state = ensureChatThreadsState();
        if (!state.chats.some((chat) => chat.id === chatId)) return;
        state.activeChatId = chatId;
        saveChatThreadsState();
        renderActiveChatConversation();
        const input = document.getElementById('chat-input');
        if (input && isAuthenticated()) input.focus();
    }

    async function startNewChat() {
        await createNewChat();
        lastAssistantResult = null;
        renderActiveChatConversation();
        setConnectionStatus(true, 'Ready');
        const input = document.getElementById('chat-input');
        if (input && isAuthenticated()) {
            input.value = '';
            input.style.height = 'auto';
            input.focus();
        }
    }

    function appendMessageRecordToActiveChat(messageRecord) {
        const activeChat = getActiveChat();
        if (!activeChat) return;
        activeChat.messages.push(messageRecord);
        touchActiveChatMeta();
        renderChatThreadList();
        updateActiveChatLabel();
    }

    async function persistMessageRecordToServer(messageRecord, chatIdOverride = null) {
        try {
            if (!chatThreadsState?.chats?.length) {
                await ensureChatThreadsStateReady();
            }
            const activeChat = chatIdOverride
                ? (ensureChatThreadsState().chats.find((chat) => chat.id === chatIdOverride) || null)
                : getActiveChat();
            const chatId = chatIdOverride || activeChat?.id;
            if (!chatId) return;
            await chatApiFetch(`/api/v1/chats/${encodeURIComponent(chatId)}/messages`, {
                method: 'POST',
                body: JSON.stringify({
                    role: messageRecord.role,
                    content: messageRecord.content,
                    rawHtml: Boolean(messageRecord.rawHtml),
                    tone: messageRecord.tone || 'normal',
                    timestamp: messageRecord.timestamp,
                }),
            });
        } catch (error) {
            console.warn('Failed to persist message to server chat memory', error);
        }
    }

    function addChatMessage(content, options = {}) {
        const messageRecord = {
            role: options.role === 'user' ? 'user' : 'assistant',
            content: String(content ?? ''),
            rawHtml: Boolean(options.rawHtml),
            tone: options.tone === 'error' ? 'error' : 'normal',
            timestamp: options.timestamp || nowIso(),
        };

        if (options.persist !== false) {
            appendMessageRecordToActiveChat(messageRecord);
        }
        if (options.persist !== false && options.serverSync !== 'none') {
            void persistMessageRecordToServer(messageRecord, options.chatId || null);
        }

        hideWelcome();
        const messageElement = renderStoredMessage(messageRecord);
        if (
            options.attachPager &&
            messageElement &&
            messageRecord.role !== 'user' &&
            !messageRecord.rawHtml &&
            messageRecord.tone !== 'error'
        ) {
            attachResponseBubblePager(messageRecord.content, messageElement);
        }
        if (options.attachServerPager && messageElement && options.apiResult) {
            appendPaginationControls(options.apiResult, messageElement);
        }
        return { record: messageRecord, element: messageElement };
    }

    function maybeWarnBackendContextDesync(apiData) {
        const activeChat = getActiveChat();
        if (!activeChat) return;
        if (backendContextDesyncWarnedChats.has(activeChat.id)) return;

        const backendHistorySize = Number(apiData?.history_size ?? 0);
        const localMessageCount = Array.isArray(activeChat.messages) ? activeChat.messages.length : 0;

        if (localMessageCount >= 4 && backendHistorySize > 0 && backendHistorySize <= 2) {
            backendContextDesyncWarnedChats.add(activeChat.id);
            addChatMessage(
                'Note: The visible chat history is available, but the server conversation memory appears to have reset. Follow-up questions may need the full context restated.',
                { role: 'assistant', persist: false }
            );
        }
    }

    async function initializeChatThreadsUi() {
        await ensureChatThreadsStateReady();
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn && !newChatBtn.dataset.bound) {
            newChatBtn.addEventListener('click', () => { void startNewChat(); });
            newChatBtn.dataset.bound = '1';
        }
        const clearChatsBtn = document.getElementById('clearChatsBtn');
        if (clearChatsBtn && !clearChatsBtn.dataset.bound) {
            clearChatsBtn.addEventListener('click', () => { void clearAllChats(); });
            clearChatsBtn.dataset.bound = '1';
        }
        renderActiveChatConversation();
    }

    function isAuthenticated() {
        try {
            return localStorage.getItem(AUTH_STORAGE_KEY) === '1' && Boolean(getStoredBearerToken());
        } catch (_error) {
            return Boolean(getStoredBearerToken());
        }
    }

    function getLockoutUntil() {
        return Number(localStorage.getItem(AUTH_LOCKOUT_KEY) || 0);
    }

    function setLockoutUntil(timestampMs) {
        if (!timestampMs || timestampMs <= 0) {
            localStorage.removeItem(AUTH_LOCKOUT_KEY);
            return;
        }
        localStorage.setItem(AUTH_LOCKOUT_KEY, String(timestampMs));
    }

    function getRemainingLockoutSeconds() {
        const until = getLockoutUntil();
        if (!until) return 0;
        return Math.max(0, Math.ceil((until - Date.now()) / 1000));
    }

    function isLockedOut() {
        return getRemainingLockoutSeconds() > 0;
    }

    function showLoginFeedback(message, tone = 'error') {
        const feedback = document.getElementById('loginFeedback');
        if (!feedback) return;
        feedback.className = `login-feedback ${tone}`;
        feedback.textContent = message || '';
    }

    function showLoginPortal() {
        const loginPortal = document.getElementById('loginPortal');
        const chatApp = document.getElementById('chatApp');
        if (chatApp) chatApp.style.display = 'none';
        if (loginPortal) loginPortal.style.display = 'block';
    }

    function showChatApp() {
        const loginPortal = document.getElementById('loginPortal');
        const chatApp = document.getElementById('chatApp');
        if (loginPortal) loginPortal.style.display = 'none';
        if (chatApp) chatApp.style.display = 'flex';
    }

    function stopLockoutTicker() {
        if (lockoutIntervalRef) {
            clearInterval(lockoutIntervalRef);
            lockoutIntervalRef = null;
        }
    }

    function startLockoutTicker() {
        stopLockoutTicker();
        const refresh = () => {
            const remaining = getRemainingLockoutSeconds();
            const loginBtn = document.getElementById('loginBtn');
            if (remaining <= 0) {
                stopLockoutTicker();
                if (loginBtn) loginBtn.disabled = false;
                showLoginFeedback('You can try logging in again.', 'warning');
                return;
            }
            if (loginBtn) loginBtn.disabled = true;
            showLoginFeedback(`Too many failed attempts. Try again in ${remaining}s.`, 'warning');
        };
        refresh();
        lockoutIntervalRef = setInterval(refresh, 1000);
    }

    function clearAuthState() {
        localStorage.removeItem(AUTH_STORAGE_KEY);
        localStorage.removeItem(AUTH_BEARER_STORAGE_KEY);
        try { sessionStorage.removeItem(AUTH_BEARER_STORAGE_KEY); } catch (_e) {}
        setLockoutUntil(0);
        loginFailedAttempts = 0;
        stopLockoutTicker();
    }

    function applyAuthGateOnLoad() {
        const usernameInput = document.getElementById('loginUsername');
        const pinInput = document.getElementById('loginPin');

        if (isAuthenticated()) {
            showChatApp();
            setConnectionStatus(true, 'Ready');
            const chatInput = document.getElementById('chat-input');
            if (chatInput) chatInput.focus();
            return;
        }

        showLoginPortal();
        setConnectionStatus(false, 'Locked');
        if (usernameInput) usernameInput.focus();
        if (pinInput) pinInput.value = '';
        if (isLockedOut()) {
            startLockoutTicker();
        } else {
            const loginBtn = document.getElementById('loginBtn');
            if (loginBtn) loginBtn.disabled = false;
        }
    }

    async function handleLoginSubmit(event) {
        event.preventDefault();

        const usernameInput = document.getElementById('loginUsername');
        const pinInput = document.getElementById('loginPin');
        const loginBtn = document.getElementById('loginBtn');
        if (!usernameInput || !pinInput || !loginBtn) return;

        if (isLockedOut()) {
            startLockoutTicker();
            return;
        }

        const username = usernameInput.value.trim();
        const pin = pinInput.value.trim();
        if (!username || !pin) {
            showLoginFeedback('Enter both username and PIN.', 'error');
            return;
        }

        loginBtn.disabled = true;
        showLoginFeedback('Signing in...', 'warning');

        try {
            const payload = await chatApiFetch('/api/v1/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, pin }),
            });
            const accessToken = typeof payload?.access_token === 'string' ? payload.access_token.trim() : '';
            if (!accessToken) {
                throw new Error('Login response did not include an access token.');
            }
            localStorage.setItem(AUTH_BEARER_STORAGE_KEY, accessToken);
            localStorage.setItem(AUTH_STORAGE_KEY, '1');
            setLockoutUntil(0);
            loginFailedAttempts = 0;
            stopLockoutTicker();
            showLoginFeedback('Login successful. Redirecting...', 'success');
            showChatApp();
            setConnectionStatus(true, 'Ready');
            pinInput.value = '';
            const chatInput = document.getElementById('chat-input');
            if (chatInput) chatInput.focus();
            return;
        } catch (error) {
            const status = Number(error?.status || 0);
            if (status === 401) {
                loginFailedAttempts += 1;
                const remainingAttempts = Math.max(0, MAX_LOGIN_ATTEMPTS - loginFailedAttempts);
                pinInput.value = '';
                pinInput.focus();

                if (remainingAttempts <= 0) {
                    setLockoutUntil(Date.now() + LOCKOUT_MS);
                    loginFailedAttempts = 0;
                    startLockoutTicker();
                    return;
                }

                showLoginFeedback(`Invalid credentials. ${remainingAttempts} attempt(s) remaining.`, 'error');
                return;
            }

            if (status === 404) {
                showLoginFeedback('Backend demo login is disabled. Enable MCP_ENABLE_DEMO_LOGIN and restart the server.', 'error');
            } else {
                showLoginFeedback(`Login failed: ${error.message || 'Unknown error'}`, 'error');
            }
            return;
        } finally {
            loginBtn.disabled = false;
        }
    }

    function logout() {
        clearAuthState();
        showLoginPortal();
        setConnectionStatus(false, 'Locked');
        const usernameInput = document.getElementById('loginUsername');
        const pinInput = document.getElementById('loginPin');
        if (usernameInput) usernameInput.focus();
        if (pinInput) pinInput.value = '';
        showLoginFeedback('You have been logged out.', 'warning');
    }

    function setConnectionStatus(online, label) {
        const indicator = document.querySelector('.status-indicator');
        const text = document.getElementById('connectionStatus');
        if (!indicator || !text) return;
        indicator.classList.toggle('offline', !online);
        text.textContent = label;
    }

    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    function handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    }

    function sendSuggestion(text) {
        document.getElementById('chat-input').value = text;
        sendMessage();
    }

    function sendPaginationCommand(command) {
        document.getElementById('chat-input').value = command;
        sendMessage();
    }

    function appendPaginationControls(apiResult, messageDiv) {
        const pagination = apiResult?.data?.pagination;
        if (!pagination || !messageDiv) return false;

        const totalPages = Number(pagination.total_pages || 1);
        const page = Number(pagination.page || 1);
        const total = Number(pagination.total || 0);
        if (totalPages <= 1) return false;

        const disablePrev = page <= 1 ? 'disabled' : '';
        const disableNext = page >= totalPages ? 'disabled' : '';
        const controlsHtml = `
            <div class="pager-wrap">
                <button class="pager-btn" ${disablePrev} onclick="sendPaginationCommand('previous page')">&larr; Back</button>
                <span class="pager-info">Page ${page} of ${totalPages} &bull; Total ${total}</span>
                <button class="pager-btn" ${disableNext} onclick="sendPaginationCommand('next page')">Next &rarr;</button>
            </div>
        `;
        insertControlsBeforeTime(messageDiv, controlsHtml, 'server-pager-controls');
        return true;
    }

    function insertControlsBeforeTime(messageDiv, controlsHtml, className = 'inline-pager-controls') {
        const contentDiv = messageDiv.querySelector('.message-content');
        if (!contentDiv) return;

        const existing = contentDiv.querySelector(`.${className}`);
        if (existing) existing.remove();

        const controls = document.createElement('div');
        controls.className = className;
        controls.innerHTML = controlsHtml;

        const timeDiv = contentDiv.querySelector('.message-time');
        if (timeDiv) {
            contentDiv.insertBefore(controls, timeDiv);
        } else {
            contentDiv.appendChild(controls);
        }
    }

    function buildNumberedListPagerModel(rawText, pageSize = 12) {
        if (typeof rawText !== 'string') return null;
        const trimmed = rawText.trim();
        if (!trimmed || trimmed.startsWith('```')) return null;

        const lines = rawText.split(/\r?\n/);
        const numberedIndexes = [];
        for (let i = 0; i < lines.length; i++) {
            if (/^\s*\d+\.\s+/.test(lines[i])) {
                numberedIndexes.push(i);
            }
        }
        if (numberedIndexes.length <= pageSize) return null;

        const first = numberedIndexes[0];
        const last = numberedIndexes[numberedIndexes.length - 1];
        const prefix = lines.slice(0, first);
        const items = lines.slice(first, last + 1).filter((line) => /^\s*\d+\.\s+/.test(line));
        const suffix = lines
            .slice(last + 1)
            .filter((line) => !/^\s*\.\.\.\s+and\s+\d+\s+more/i.test(line.trim()));

        if (items.length <= pageSize) return null;

        const pages = [];
        for (let i = 0; i < items.length; i += pageSize) {
            pages.push(items.slice(i, i + pageSize));
        }

        return {
            prefix,
            suffix,
            pages,
            total: items.length,
            pageSize,
        };
    }

    function renderResponseBubblePage(pagerId) {
        const state = responseBubblePagers.get(pagerId);
        if (!state) return;

        const { messageDiv, model, page } = state;
        const contentDiv = messageDiv.querySelector('.message-content');
        if (!contentDiv) return;

        const timeText = messageDiv.dataset.timeLabel || formatTime(new Date());
        const pageItems = model.pages[page] || [];
        const start = page * model.pageSize + 1;
        const end = Math.min(start + pageItems.length - 1, model.total);

        const pageLines = [
            ...model.prefix,
            '',
            ...pageItems,
            '',
            `Showing ${start}-${end} of ${model.total}`,
            ...model.suffix,
        ];

        contentDiv.innerHTML = OutputLayer.formatResponse(pageLines.join('\n'));

        const controlsHtml = `
            <div class="pager-wrap">
                <button class="pager-btn" ${page <= 0 ? 'disabled' : ''} onclick="changeResponseBubblePage('${pagerId}', -1)">&larr; Back</button>
                <span class="pager-info">Page ${page + 1} of ${model.pages.length}</span>
                <button class="pager-btn" ${page >= model.pages.length - 1 ? 'disabled' : ''} onclick="changeResponseBubblePage('${pagerId}', 1)">Next &rarr;</button>
            </div>
        `;
        insertControlsBeforeTime(messageDiv, controlsHtml, 'inline-pager-controls');

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = timeText;
        contentDiv.appendChild(timeDiv);
    }

    function attachResponseBubblePager(rawText, messageDiv) {
        const model = buildNumberedListPagerModel(rawText);
        if (!model) return;

        const pagerId = `rsp_${++responseBubblePagerCounter}`;
        responseBubblePagers.set(pagerId, {
            messageDiv,
            model,
            page: 0,
        });
        renderResponseBubblePage(pagerId);
    }

    function changeResponseBubblePage(pagerId, delta) {
        const state = responseBubblePagers.get(pagerId);
        if (!state) return;
        const nextPage = state.page + delta;
        if (nextPage < 0 || nextPage >= state.model.pages.length) return;
        state.page = nextPage;
        renderResponseBubblePage(pagerId);
        scrollToBottom();
    }

    function startPersonnelSearch() {
        hideWelcome();
        const input = document.getElementById('chat-input');
        const helperMessage = [
            "To search personnel, include at least one field:",
            "- Name",
            "- User ID",
            "- Badge number",
            "- Mobile",
            "- Email",
            "",
            "Examples:",
            "- Find personnel with name <person_name>",
            "- Search personnel by badge no <badge_number>",
            "- Get personnel details for mobile <mobile_number>",
        ].join('\n');

        addChatMessage(helperMessage, { role: 'assistant' });
        input.value = '';
        input.placeholder = "Type your question here...";
        input.focus();
        scrollToBottom();
    }

    function formatTime(date) {
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }

    function createMessage(content, isUser = false, useRawHtml = false, options = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = isUser
            ? '<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = useRawHtml ? content : OutputLayer.formatResponse(content);

        if (options.tone === 'error') {
            contentDiv.style.background = 'var(--danger-bg)';
            contentDiv.style.borderColor = 'var(--danger-border)';
            contentDiv.style.color = 'var(--danger-text)';
        }

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        const timestamp = options.timestamp ? new Date(options.timestamp) : new Date();
        timeDiv.textContent = formatTime(timestamp);
        if (options.tone === 'error') {
            timeDiv.style.color = 'var(--danger-text)';
        }
        messageDiv.dataset.timeLabel = timeDiv.textContent;
        contentDiv.appendChild(timeDiv);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);

        return messageDiv;
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message bot';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
            </div>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        document.getElementById('chatMessages').appendChild(indicator);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    function scrollToBottom() {
        const messages = document.getElementById('chatMessages');
        requestAnimationFrame(() => {
            messages.scrollTop = messages.scrollHeight;
        });
    }

    function hideWelcome() {
        const welcome = document.getElementById('welcomeMessage');
        if (welcome) welcome.style.display = 'none';
    }

    async function sendMessage() {
        if (!isAuthenticated()) {
            showLoginPortal();
            setConnectionStatus(false, 'Locked');
            return;
        }

        await ensureChatThreadsStateReady();

        const input = document.getElementById('chat-input');
        const query = input.value.trim();

        if (!query) return;

        hideWelcome();
        const outputCommand = OutputLayer.detectOutputCommand(query);
        const isMarkupLike = /<\s*\/?\s*[a-z][^>]*>/i.test(query);
        const willCallApi = !outputCommand && !isMarkupLike;
        const userMessage = addChatMessage(query, { role: 'user', serverSync: willCallApi ? 'none' : 'append' });
        if (outputCommand) {
            if (lastAssistantResult == null) {
                addChatMessage('No previous results to format. Please run a search first, then ask for JSON/table/tree/chart/download.', { role: 'assistant' });
                input.value = '';
                input.style.height = 'auto';
                scrollToBottom();
                input.focus();
                return;
            }
            if (outputCommand.kind === 'visual') {
                const chartHtml = OutputLayer.renderVisualizationFromLast(lastAssistantResult, outputCommand.format || 'bar');
                if (!chartHtml) {
                    addChatMessage('I can build charts from numeric distribution data. Please run a query like personnel distribution first, then ask for a visual.', { role: 'assistant' });
                } else {
                    addChatMessage(chartHtml, { role: 'assistant', rawHtml: true });
                }
            } else if (outputCommand.kind === 'table') {
                const tableHtml = OutputLayer.renderTableFromLast(lastAssistantResult);
                if (!tableHtml) {
                    addChatMessage('I could not build a table from the previous result. Try asking for JSON first, then ask for table.', { role: 'assistant' });
                } else {
                    addChatMessage(tableHtml, { role: 'assistant', rawHtml: true });
                }
            } else {
                const rendered = OutputLayer.renderFromLast(lastAssistantResult, outputCommand.format);
                if (!rendered) {
                    addChatMessage('No previous result is available to format or download yet.', { role: 'assistant' });
                    scrollToBottom();
                    input.value = '';
                    input.style.height = 'auto';
                    input.focus();
                    return;
                }

                if (outputCommand.kind === 'render') {
                    const fenced = outputCommand.format === 'text'
                        ? rendered
                        : `\`\`\`${outputCommand.format === 'tree' ? 'text' : outputCommand.format}\n${rendered}\n\`\`\``;
                    addChatMessage(fenced, { role: 'assistant' });
                } else {
                    const ext = outputCommand.format === 'json' ? 'json' : 'txt';
                    const contentType = outputCommand.format === 'json' ? 'application/json' : 'text/plain';
                    OutputLayer.triggerDownload({
                        download: {
                            content: rendered,
                            filename: `query_result.${ext}`,
                            content_type: contentType,
                        },
                    });
                    addChatMessage(`Downloaded the last result as ${ext.toUpperCase()}.`, { role: 'assistant' });
                }
            }

            input.value = '';
            input.style.height = 'auto';
            scrollToBottom();
            input.focus();
            return;
        }

        // Block markup/tag-like input locally to avoid confusing/stale-looking
        // responses and keep the chat strictly plain-text.
        if (isMarkupLike) {
            addChatMessage('Please enter a plain-text query (no HTML/script tags). For example: "List units in Guntur district".', { role: 'assistant', tone: 'error' });
            input.value = '';
            input.style.height = 'auto';
            scrollToBottom();
            input.focus();
            return;
        }

        input.value = '';
        input.style.height = 'auto';

        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;

        showTypingIndicator();
        scrollToBottom();
        setConnectionStatus(true, 'Sending...');

        try {
            const activeChat = getActiveChat();
            const response = await fetch(`${config.apiUrl}/api/v1/ask`, {
                method: 'POST',
                headers: buildApiHeaders(),
                body: JSON.stringify({
                    query: query,
                    output_format: 'auto',
                    allow_download: null,
                    chat_id: activeChat?.id || null,
                    session_id: getActiveChatSessionId(),
                }),
            });

            removeTypingIndicator();

            if (!response.ok) {
                let serverMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errorPayload = await response.json();
                    const detail = errorPayload?.detail;
                    if (typeof detail === 'string' && detail.trim()) {
                        serverMessage = detail.trim();
                    } else if (detail && typeof detail === 'object') {
                        const code = detail.code || detail.error?.code;
                        const detailMessage = detail.message || detail.error?.message;
                        const userAction = detail.user_action || detail.error?.user_action;
                        const requestId = detail.details?.request_id;
                        if (detailMessage) {
                            const parts = [];
                            if (code) {
                                parts.push(`[${code}]`);
                            }
                            parts.push(detailMessage);
                            if (requestId) {
                                parts.push(`(Request ID: ${requestId})`);
                            }
                            if (userAction) {
                                parts.push(userAction);
                            }
                            serverMessage = parts.join(' ');
                        }
                    } else if (errorPayload?.error?.message) {
                        serverMessage = errorPayload.error.message;
                    }
                } catch (_ignored) {
                    // Keep default HTTP status-based message when response is not JSON.
                }
                const httpError = new Error(serverMessage);
                httpError.status = response.status;
                throw httpError;
            }

            const data = await response.json();
            lastAssistantResult = data;
            setActiveChatLastAssistantResult(data);

            let responseText = '';
            if (data.output && data.output.rendered) {
                if (data.output.format === 'json') {
                    responseText = `\`\`\`json\n${data.output.rendered}\n\`\`\``;
                } else if (data.output.format === 'tree') {
                    responseText = `\`\`\`text\n${data.output.rendered}\n\`\`\``;
                } else {
                    responseText = data.output.rendered;
                }
            } else if (data.response) {
                responseText = data.response;
            } else if (data.data && typeof data.data === 'object') {
                responseText = JSON.stringify(data.data, null, 2);
            } else {
                responseText = JSON.stringify(data, null, 2);
            }

            const responseMessage = addChatMessage(responseText, { role: 'assistant', serverSync: 'none' }).element;
            const hasServerPager = appendPaginationControls(data, responseMessage);
            if (!hasServerPager) {
                attachResponseBubblePager(responseText, responseMessage);
            }
            maybeWarnBackendContextDesync(data);
            if (data.output && data.output.download && data.output.download.content) {
                OutputLayer.triggerDownload(data.output);
            }
            setConnectionStatus(true, 'Connected');
        } catch (error) {
            removeTypingIndicator();
            const activeChat = getActiveChat();
            const activeChatId = activeChat?.id || null;
            if (willCallApi && userMessage?.record && activeChatId) {
                await persistMessageRecordToServer(userMessage.record, activeChatId);
            }

            let errorMessage = 'Sorry, I encountered an error while processing your request.';
            if (error.message.includes('Failed to fetch')) {
                errorMessage = `Unable to connect to the server at ${config.apiUrl}. Please check if the server is running.`;
            } else {
                errorMessage += ` Error: ${error.message}`;
            }

            addChatMessage(errorMessage, { role: 'assistant', tone: 'error' });
            setConnectionStatus(false, 'Disconnected');
        }

        sendBtn.disabled = false;
        scrollToBottom();
        input.focus();
    }

    let appInitialized = false;
    async function initializeApp() {
        if (appInitialized) return;
        appInitialized = true;

        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', handleLoginSubmit);
        }

        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', logout);
        }

        await initializeChatThreadsUi();
        applyAuthGateOnLoad();
    }

    window.autoResize = autoResize;
    window.handleKeyDown = handleKeyDown;
    window.sendSuggestion = sendSuggestion;
    window.sendPaginationCommand = sendPaginationCommand;
    window.changeResponseBubblePage = changeResponseBubblePage;
    window.startPersonnelSearch = startPersonnelSearch;
    window.startNewChat = startNewChat;
    window.sendMessage = sendMessage;
    window.logout = logout;
    window.handleLoginSubmit = handleLoginSubmit;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => { void initializeApp(); }, { once: true });
        window.addEventListener('load', () => { void initializeApp(); }, { once: true });
    } else {
        void initializeApp();
    }
})();
