import {
  AlertCircle,
  ArrowUpRight,
  Bot,
  ChevronDown,
  ChevronRight,
  Database,
  FileUp,
  FileText,
  FolderPlus,
  Loader2,
  LogOut,
  Menu,
  Pencil,
  Plus,
  RefreshCcw,
  Send,
  Sparkles,
  Trash2,
  X
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  askMoneyAgent,
  ApiError,
  buildDomain,
  ChatResponse,
  createDomain,
  createSession,
  deleteDomain,
  deleteDomainFile,
  deleteSession,
  DomainFile,
  DomainInfo,
  Evidence,
  fetchDomainFiles,
  fetchCurrentUser,
  fetchDomains,
  fetchSession,
  fetchSessions,
  login,
  logout,
  renameSession,
  replaceDomainFile,
  SessionInfo,
  uploadDomainFile
} from "./lib/api";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  meta?: ChatResponse;
  error?: string;
};

const EXAMPLE_QUESTIONS = [
  { label: "保险条款", question: "请比较平安e生保和太保团体百万医疗的免赔额规则。" },
  { label: "研究报告", question: "东方甄选相关研报中，GMV 和自营产品表现有什么变化？" },
  { label: "金融合同", question: "请根据金融合同说明债券应计利息的计算逻辑。" }
];

const DOMAIN_LABELS: Record<string, string> = {
  financial_contracts: "金融合同",
  financial_reports: "财报公告",
  insurance: "保险条款",
  regulatory: "监管文件",
  research: "研究报告"
};

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function formatNumber(value: number | undefined) {
  return new Intl.NumberFormat("zh-CN").format(value || 0);
}

function domainLabel(domain: DomainInfo) {
  return DOMAIN_LABELS[domain.domain] || domain.name || domain.domain;
}

function Sidebar({
  username,
  onLogout,
  domains,
  selectedDomain,
  onDomainChange,
  loading,
  onReload,
  mobileOpen,
  onClose,
  sessions,
  activeSessionId,
  sessionsLoading,
  disabled,
  onSelectSession,
  onCreateSession,
  onRenameSession,
  onDeleteSession,
  onDomainsChanged,
  onDomainRemoved
}: {
  username: string;
  onLogout: () => void;
  domains: DomainInfo[];
  selectedDomain: string;
  onDomainChange: (domain: string) => void;
  loading: boolean;
  onReload: () => void;
  mobileOpen: boolean;
  onClose: () => void;
  sessions: SessionInfo[];
  activeSessionId: string;
  sessionsLoading: boolean;
  disabled: boolean;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onRenameSession: (session: SessionInfo) => void;
  onDeleteSession: (session: SessionInfo) => void;
  onDomainsChanged: (selectDomain?: string) => Promise<void>;
  onDomainRemoved: (domain: string) => Promise<void>;
}) {
  const [showManager, setShowManager] = useState(false);
  const [domainName, setDomainName] = useState("");
  const [domainKey, setDomainKey] = useState("");
  const [splitMode, setSplitMode] = useState("heading");
  const [managedDomain, setManagedDomain] = useState("");
  const [domainBusy, setDomainBusy] = useState("");
  const [domainError, setDomainError] = useState("");
  const [domainNotice, setDomainNotice] = useState("");
  const [domainFiles, setDomainFiles] = useState<DomainFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const customDomains = domains.filter((domain) => !DOMAIN_LABELS[domain.domain]);
  const targetDomain = customDomains.some((domain) => domain.domain === managedDomain)
    ? managedDomain
    : customDomains[0]?.domain || "";
  const targetInfo = customDomains.find((domain) => domain.domain === targetDomain);

  useEffect(() => {
    if (!showManager) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setShowManager(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showManager]);

  useEffect(() => {
    if (!showManager || !targetDomain) {
      setDomainFiles([]);
      return;
    }
    let current = true;
    setFilesLoading(true);
    fetchDomainFiles(targetDomain)
      .then((result) => {
        if (current) setDomainFiles(result.files);
      })
      .catch((caught) => {
        if (current) setDomainError(caught instanceof Error ? caught.message : "无法读取文件列表");
      })
      .finally(() => {
        if (current) setFilesLoading(false);
      });
    return () => { current = false; };
  }, [showManager, targetDomain]);

  async function refreshFiles(domain: string) {
    const result = await fetchDomainFiles(domain);
    setDomainFiles(result.files);
  }

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setDomainBusy("创建中");
    setDomainError("");
    setDomainNotice("");
    try {
      const result = await createDomain({ domain: domainKey.trim(), name: domainName.trim(), split_mode: splitMode });
      await onDomainsChanged();
      setManagedDomain(result.domain);
      setDomainFiles([]);
      setDomainName("");
      setDomainKey("");
      setDomainNotice("领域已创建，请上传文件后构建。");
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "创建领域失败");
    } finally {
      setDomainBusy("");
    }
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem("domain-files") as HTMLInputElement | null;
    if (!targetDomain || !input?.files?.length) return;
    setDomainBusy("上传中");
    setDomainError("");
    setDomainNotice("");
    try {
      const files = Array.from(input.files);
      for (const file of files) await uploadDomainFile(targetDomain, file);
      input.value = "";
      await onDomainsChanged();
      await refreshFiles(targetDomain);
      setDomainNotice(`已上传 ${files.length} 个文件，请点击构建索引。`);
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "上传文件失败");
    } finally {
      try {
        await onDomainsChanged();
        await refreshFiles(targetDomain);
      } catch {
        // Preserve the original upload error.
      }
      setDomainBusy("");
    }
  }

  async function build() {
    if (!targetDomain) return;
    setDomainBusy("构建中");
    setDomainError("");
    setDomainNotice("");
    try {
      const result = await buildDomain(targetDomain);
      await onDomainsChanged(targetDomain);
      setDomainNotice(`构建完成：${result.doc_count} 份文档，${result.page_count} 个页面。`);
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "构建索引失败");
    } finally {
      setDomainBusy("");
    }
  }

  async function replaceFile(filename: string, file: File) {
    if (!targetDomain || !window.confirm(`确定用“${file.name}”替换“${filename}”吗？替换后需要重新构建索引。`)) return;
    setDomainBusy("替换中");
    setDomainError("");
    setDomainNotice("");
    try {
      await replaceDomainFile(targetDomain, filename, file);
      await onDomainsChanged();
      await refreshFiles(targetDomain);
      setDomainNotice(`已替换 ${filename}，请重新构建索引。`);
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "替换文件失败");
    } finally {
      setDomainBusy("");
    }
  }

  async function removeFile(filename: string) {
    if (!targetDomain || !window.confirm(`确定删除“${filename}”吗？删除后需要重新构建索引。`)) return;
    setDomainBusy("删除中");
    setDomainError("");
    setDomainNotice("");
    try {
      await deleteDomainFile(targetDomain, filename);
      await onDomainsChanged();
      await refreshFiles(targetDomain);
      setDomainNotice(`已删除 ${filename}，请重新构建索引。`);
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "删除文件失败");
    } finally {
      setDomainBusy("");
    }
  }

  async function removeDomain() {
    if (!targetDomain || !window.confirm(`确定删除“${targetInfo?.name || targetDomain}”及其上传文件和索引吗？该操作无法撤销。`)) return;
    setDomainBusy("删除领域中");
    setDomainError("");
    setDomainNotice("");
    try {
      await deleteDomain(targetDomain);
      await onDomainRemoved(targetDomain);
      setManagedDomain("");
      setDomainFiles([]);
      setDomainNotice("领域及其文件和索引已删除。");
    } catch (caught) {
      setDomainError(caught instanceof Error ? caught.message : "删除领域失败");
    } finally {
      setDomainBusy("");
    }
  }

  return (
    <aside className={cn(
      "workspace-sidebar fixed inset-y-0 left-0 z-40 flex w-[min(18rem,calc(100vw-3rem))] flex-col shadow-xl lg:relative lg:z-auto lg:h-screen lg:w-[268px] lg:shrink-0 lg:shadow-none",
      !mobileOpen && "hidden lg:flex"
    )}>
      <div className="flex h-[76px] shrink-0 items-center justify-between border-b border-white/10 px-5">
        <div className="flex min-w-0 items-center gap-3">
          <span className="brand-mark" aria-hidden="true">
            M<span>.</span>
          </span>
          <div className="min-w-0">
            <div className="text-[15px] font-semibold leading-5 text-white">MoneyAgent</div>
            <div className="mt-0.5 text-[11px] text-white/45">RESEARCH WORKSPACE</div>
          </div>
        </div>
        <button type="button" onClick={onClose} aria-label="关闭会话列表" title="关闭会话列表"
          className="sidebar-icon-button lg:hidden"><X className="h-4 w-4" /></button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="px-4 pt-5">
          <button type="button" onClick={onCreateSession} disabled={disabled}
            className="sidebar-new-button">
            <Plus className="h-4 w-4" /> 新建研究
          </button>
        </div>
        <section className="flex min-h-0 flex-1 flex-col px-3 pt-7">
          <div className="mb-3 flex items-center justify-between px-3">
            <h2 className="sidebar-label">对话记录</h2>
            <button
              type="button"
              onClick={onCreateSession}
              disabled={disabled}
              className="sidebar-icon-button"
              aria-label="新建会话"
              title="新建会话"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto pb-3">
            {sessionsLoading && sessions.length === 0 ? (
              Array.from({ length: 3 }).map((_, index) => (
                <div key={index} className="mx-1 mb-2 h-12 animate-pulse rounded-md bg-white/10" />
              ))
            ) : sessions.length === 0 ? (
              <div className="px-3 py-6 text-xs text-white/50">暂无会话，开始一项新的研究。</div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={cn(
                    "sidebar-session group flex items-center gap-1 rounded-md border-l-2 border-transparent px-1 py-1 transition-colors",
                    activeSessionId === session.session_id && "sidebar-session-active"
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelectSession(session.session_id)}
                    disabled={disabled}
                    className="min-w-0 flex-1 rounded-sm px-2 py-1.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70 disabled:cursor-not-allowed"
                  >
                    <div className="truncate text-[13px] font-medium text-white/90">{session.title || "新会话"}</div>
                    <div className="mt-1 text-[11px] text-white/45">{session.turn_count} 轮对话</div>
                  </button>
                  <div className="flex shrink-0 lg:opacity-0 lg:group-hover:opacity-100 lg:group-focus-within:opacity-100">
                    <button
                      type="button"
                      onClick={() => onRenameSession(session)}
                      disabled={disabled}
                      className="sidebar-icon-button h-7 w-7"
                      aria-label={`重命名 ${session.title || "会话"}`}
                      title="重命名"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteSession(session)}
                      disabled={disabled}
                      className="sidebar-icon-button h-7 w-7 hover:!text-rose-300"
                      aria-label={`删除 ${session.title || "会话"}`}
                      title="删除"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="shrink-0 border-t border-white/10 px-3 py-4">
          <div className="mb-2 flex items-center justify-between px-3">
            <h2 className="sidebar-label">研究资料</h2>
            <button
              type="button"
              onClick={() => setShowManager((value) => !value)}
              aria-expanded={showManager}
              className="sidebar-icon-button"
              aria-label="管理知识域"
              title="管理知识域"
            >
              <FolderPlus className="h-4 w-4" />
            </button>
          </div>
          {showManager && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/45 p-3 sm:p-6" onMouseDown={(event) => {
              if (event.target === event.currentTarget) setShowManager(false);
            }}>
              <div role="dialog" aria-modal="true" aria-labelledby="domain-manager-title"
                className="flex max-h-[min(90dvh,780px)] w-full max-w-3xl flex-col overflow-hidden rounded-lg border bg-card text-foreground shadow-xl">
                <div className="flex shrink-0 items-center justify-between border-b px-5 py-4">
                  <div>
                    <h2 id="domain-manager-title" className="text-base font-semibold">管理知识域</h2>
                    <p className="mt-1 text-xs text-muted-foreground">上传文档并构建检索索引</p>
                  </div>
                  <button type="button" onClick={() => setShowManager(false)} className="icon-button"
                    aria-label="关闭知识域管理" title="关闭知识域管理"><X className="h-4 w-4" /></button>
                </div>
                <div className="min-h-0 overflow-y-auto p-5">
                  <div className="grid gap-7 md:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
              <form onSubmit={(event) => void create(event)} className="space-y-3 text-xs">
                <div className="text-sm font-semibold">创建领域</div>
                <label htmlFor="domain-name" className="block text-muted-foreground">领域名称</label>
                <input id="domain-name" value={domainName} onChange={(event) => setDomainName(event.target.value)}
                  maxLength={60} required placeholder="例如：新能源行业"
                  className="field" />
                <label htmlFor="domain-key" className="block text-muted-foreground">领域标识（英文）</label>
                <input id="domain-key" value={domainKey} onChange={(event) => setDomainKey(event.target.value)}
                  required pattern="[a-z][a-z0-9_-]{2,39}" placeholder="例如：new_energy"
                  className="field" />
                <label htmlFor="split-mode" className="block text-muted-foreground">切分方式</label>
                <select id="split-mode" value={splitMode} onChange={(event) => setSplitMode(event.target.value)}
                  className="field">
                  <option value="heading">Markdown 标题</option>
                  <option value="heading_plus_numbered">标题与编号条款</option>
                  <option value="block">段落</option>
                </select>
                <button type="submit" disabled={!!domainBusy || disabled}
                  className="secondary-button">
                  创建
                </button>
              </form>
              {customDomains.length > 0 && (
                <div className="space-y-3 border-t pt-5 text-xs md:border-l md:border-t-0 md:pl-7 md:pt-0">
                  <div className="text-sm font-semibold">上传与构建</div>
                  <label htmlFor="managed-domain" className="block text-muted-foreground">目标领域</label>
                  <select id="managed-domain" value={targetDomain} onChange={(event) => setManagedDomain(event.target.value)}
                    className="field">
                    {customDomains.map((domain) => (
                      <option key={domain.domain} value={domain.domain}>{domainLabel(domain)}</option>
                    ))}
                  </select>
                  <form onSubmit={(event) => void upload(event)} className="space-y-2">
                    <label htmlFor="domain-files" className="block text-muted-foreground">UTF-8 Markdown 或文本文件（每个不超过 8 MB）</label>
                    <input id="domain-files" name="domain-files" type="file" accept=".md,.txt" multiple required
                      className="block w-full min-w-0 text-xs file:mr-2 file:rounded-md file:border file:bg-background file:px-2 file:py-1.5 file:text-xs" />
                    <div className="flex items-center gap-2">
                      <button type="submit" disabled={!!domainBusy || disabled}
                        className="secondary-button">
                        上传
                      </button>
                      <button type="button" onClick={() => void build()}
                        disabled={!!domainBusy || disabled || !(customDomains.find((domain) => domain.domain === targetDomain)?.upload_count)}
                        className="primary-button">
                        构建索引
                      </button>
                    </div>
                  </form>
                  <div className="space-y-2 border-t pt-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium">文件（{domainFiles.length}）</span>
                      {targetInfo?.last_built_at && (
                        <time className="text-[11px] text-muted-foreground" dateTime={targetInfo.last_built_at}>
                          上次构建：{new Date(targetInfo.last_built_at).toLocaleString("zh-CN")}
                        </time>
                      )}
                    </div>
                    {targetInfo?.status === "needs_build" && (
                      <p role="status" className="text-xs leading-5 text-muted-foreground">
                        文件已更改，重新构建前检索仍使用上次成功的索引。
                      </p>
                    )}
                    {filesLoading ? (
                      <div className="h-10 animate-pulse rounded-md bg-muted/50" />
                    ) : domainFiles.length === 0 ? (
                      <p className="text-xs text-muted-foreground">暂无文件</p>
                    ) : (
                      <div className="max-h-44 space-y-1 overflow-y-auto">
                        {domainFiles.map((item) => (
                          <div key={item.filename} className="flex min-w-0 items-center gap-1 border-b py-1.5">
                            <div className="min-w-0 flex-1" title={item.filename}>
                              <div className="truncate text-xs font-medium">{item.filename}</div>
                              <div className="text-[11px] text-muted-foreground">{formatNumber(item.size_bytes)} B</div>
                            </div>
                            <label
                              title={`替换 ${item.filename}`}
                              className={cn(
                                "relative inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md hover:bg-muted focus-within:ring-2 focus-within:ring-ring",
                                (!!domainBusy || disabled) && "pointer-events-none opacity-50"
                              )}
                            >
                              <FileUp className="h-4 w-4" />
                              <input type="file" accept=".md,.txt" disabled={!!domainBusy || disabled}
                                aria-label={`替换 ${item.filename}`}
                                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                                onChange={(event) => {
                                  const replacement = event.target.files?.[0];
                                  if (replacement) void replaceFile(item.filename, replacement);
                                  event.target.value = "";
                                }} />
                            </label>
                            <button type="button" onClick={() => void removeFile(item.filename)}
                              disabled={!!domainBusy || disabled}
                              title={`删除 ${item.filename}`}
                              aria-label={`删除 ${item.filename}`}
                              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <button type="button" onClick={() => void removeDomain()} disabled={!!domainBusy || disabled}
                    className="secondary-button border-destructive/30 text-destructive hover:bg-destructive/10">
                    删除领域
                  </button>
                </div>
              )}
                  </div>
              {domainBusy && <div role="status" className="text-muted-foreground">{domainBusy}，请稍候…</div>}
              {domainError && <div role="alert" className="text-destructive">{domainError}</div>}
              {domainNotice && <div role="status" className="text-muted-foreground">{domainNotice}</div>}
                </div>
              </div>
            </div>
          )}
          <div className="max-h-[32vh] space-y-0.5 overflow-y-auto lg:max-h-[37vh]">
            {loading && domains.length === 0
              ? Array.from({ length: 3 }).map((_, index) => (
                  <div key={index} className="h-10 animate-pulse rounded-md bg-white/10" />
                ))
              : domains.map((domain) => (
                  <button
                    key={domain.domain}
                    type="button"
                    onClick={() => onDomainChange(domain.page_index_exists ? domain.domain : selectedDomain)}
                    disabled={!domain.page_index_exists}
                    className={cn(
                      "sidebar-domain flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-[13px] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70 disabled:cursor-not-allowed disabled:opacity-50",
                      selectedDomain === domain.domain && "sidebar-domain-active font-medium"
                    )}
                  >
                    <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", domain.status === "needs_build" ? "bg-amber-500" : domain.page_index_exists ? "bg-emerald-600" : "bg-muted-foreground/50")} />
                    <span className="min-w-0 flex-1 truncate text-white/80" title={domainLabel(domain)}>{domainLabel(domain)}</span>
                    <span className="shrink-0 text-[11px] font-normal text-white/45">
                      {domain.status === "needs_build" ? "待重建" : domain.page_index_exists ? `${formatNumber(domain.doc_count)} 文档` : "待构建"}
                    </span>
                  </button>
                ))}
          </div>
        </section>
        <div className="flex shrink-0 items-center justify-between border-t border-white/10 px-5 py-3 text-xs text-white/70">
          <span className="min-w-0 truncate" title={username}>{username}</span>
          <button type="button" onClick={onLogout} className="sidebar-icon-button" aria-label="退出登录" title="退出登录">
            <LogOut className="h-4 w-4" />
          </button>
          <button type="button" onClick={onReload} className="sidebar-icon-button" aria-label="刷新知识域" title="刷新知识域">
            <RefreshCcw className={cn("h-4 w-4", loading && "animate-spin")} />
          </button>
        </div>
      </div>
    </aside>
  );
}

function MessageBubble({ message, onShowEvidence }: { message: Message; onShowEvidence: (response: ChatResponse) => void }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={cn("research-turn flex min-w-0 gap-4 py-7 sm:gap-6", isAssistant ? "research-turn-answer" : "research-turn-query")}>
      <div className={cn("turn-index mt-0.5 shrink-0", isAssistant ? "turn-index-agent" : "turn-index-user")}>
        {isAssistant ? <Sparkles className="h-[18px] w-[18px]" /> : <span className="font-mono text-xs">Q</span>}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-3 flex items-center gap-3">
          <span className="text-xs font-semibold">{isAssistant ? "研究结论" : "研究问题"}</span>
          {isAssistant && <span className="h-px w-7 bg-primary/50" />}
        </div>
        <div className={cn("whitespace-pre-wrap break-words text-[14px] leading-[1.9]", isAssistant ? "text-foreground" : "font-medium text-foreground")}>{message.content}</div>
        {message.error && <p role="alert" className="mt-2 text-xs text-destructive">{message.error}</p>}
        {message.meta && (
          <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 border-t pt-4 text-xs text-muted-foreground">
            <span>范围 / {message.meta.domain || "自动识别"}</span>
            <span>引用 / {message.meta.doc_ids.length} 份文档</span>
            <button type="button" onClick={() => {
              if (message.meta) onShowEvidence(message.meta);
            }}
              className="ml-auto inline-flex items-center gap-1 font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              查看来源 <ArrowUpRight className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  loading
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
}) {
  function submit(event: FormEvent) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={submit} className="composer-region shrink-0 border-t px-4 py-3 sm:px-8 sm:py-5">
      <div className="mx-auto max-w-[780px]">
        <div className="composer-box">
        <label htmlFor="query" className="sr-only">
          输入金融问题
        </label>
        <textarea
          id="query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          rows={2}
          placeholder="输入你的研究问题…"
          className="max-h-44 min-h-16 w-full resize-y bg-transparent px-1 py-1 text-sm leading-6 outline-none placeholder:text-muted-foreground"
          disabled={loading}
        />
        <div className="flex items-center justify-between border-t pt-2">
          <span className="hidden text-xs text-muted-foreground sm:inline">基于知识库检索 · 支持连续追问</span>
          <button
            type="submit"
            disabled={loading || !value.trim()}
            className="send-button"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            发送
          </button>
        </div>
        </div>
      </div>
    </form>
  );
}

function EvidencePanel({ activeResponse, loading, open, onClose }: { activeResponse: ChatResponse | null; loading: boolean; open: boolean; onClose: () => void }) {
  const evidence = activeResponse?.evidence || [];
  const facts = activeResponse?.facts || [];
  const questionInfo = activeResponse?.question_info || [];
  return (
    <aside className={cn(
      "evidence-panel fixed inset-y-0 right-0 z-40 w-[min(25rem,100vw)] border-l shadow-xl lg:relative lg:z-auto lg:h-screen lg:w-80 lg:shrink-0 lg:shadow-none xl:w-[23rem]",
      !open && "hidden"
    )}>
      <div className="flex h-full min-h-0 flex-col">
        <div className="flex h-[76px] shrink-0 items-center justify-between border-b px-5">
          <div>
            <div className="text-sm font-semibold">资料来源</div>
            <p className="mt-1 text-xs text-muted-foreground">SOURCE INSPECTOR</p>
          </div>
          <button type="button" onClick={onClose} className="icon-button" aria-label="关闭证据" title="关闭证据">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-5">
          {loading && (
            <div className="space-y-2">
              <div className="h-20 animate-pulse rounded-lg border bg-muted/50" />
              <div className="h-20 animate-pulse rounded-lg border bg-muted/50" />
            </div>
          )}
          {!loading && !activeResponse && (
            <div className="border-l-2 border-primary/30 pl-3 text-sm">
              <div className="font-medium">暂无可展示的来源</div>
              <p className="mt-1 leading-5 text-muted-foreground">新一轮问答完成后可查看引用；历史会话暂不恢复旧证据。</p>
            </div>
          )}
          {activeResponse && (
            <>
              <div className="grid grid-cols-2 gap-3 border-b pb-5 text-sm">
                <div className="min-w-0">
                  <div className="text-xs text-muted-foreground">知识域</div>
                  <div className="mt-1 truncate font-medium">{activeResponse.domain || "自动识别"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Token 使用</div>
                  <div className="mt-1 font-medium">{formatNumber(activeResponse.token_usage.total_tokens)}</div>
                </div>
              </div>
              <EvidenceList evidence={evidence} facts={facts} questionInfo={questionInfo} />
              <details className="border-t pt-4 text-xs">
                <summary className="flex cursor-pointer items-center gap-2 font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  <Database className="h-3.5 w-3.5" />
                  调试文件
                </summary>
                <div className="mt-3 space-y-1 break-all text-muted-foreground">
                  {Object.entries(activeResponse.debug_files || {}).map(([key, value]) => (
                    <div key={key}>
                      {key}: {value}
                    </div>
                  ))}
                </div>
              </details>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}

function EvidenceList({
  evidence,
  facts,
  questionInfo
}: {
  evidence: Evidence[];
  facts: ChatResponse["facts"];
  questionInfo: ChatResponse["question_info"];
}) {
  if (!evidence.length && !facts.length && !questionInfo.length) {
    return (
      <div className="border-l-2 border-border pl-3 text-sm">
        <div className="font-medium">没有返回证据片段</div>
        <p className="mt-1 text-muted-foreground">可能是问答失败、证据为空，或当前输出没有包含 evidence summary。</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {evidence.length > 0 && (
        <section className="space-y-2">
          <div className="section-label">原文摘录 / {evidence.length}</div>
      {evidence.map((item, index) => (
        <div key={`${item.doc_id}-${item.page_id}-${index}`} className="source-item border-b py-4 last:border-0">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="source-number">{String(index + 1).padStart(2, "0")}</span>
            <span>文档 {item.doc_id || "未知"}</span>
            <span>页面 {item.page_id || item.evidence_id || "未知"}</span>
          </div>
          {item.heading && <div className="mt-2 text-xs font-medium">{item.heading}</div>}
          <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">{item.quote || "无原文摘录"}</p>
        </div>
      ))}
        </section>
      )}
      {facts.length > 0 && (
        <section className="space-y-2">
          <div className="section-label">回答要点 · {facts.length}</div>
          {facts.map((item, index) => (
            <div key={`${item.slot}-${index}`} className="border-b py-3 last:border-0">
              <div className="text-xs font-medium">{item.slot}</div>
              <p className="mt-2 break-words text-xs leading-5 text-muted-foreground">{item.value}</p>
              <div className="mt-2 text-[11px] text-muted-foreground">
                来源：{item.evidence_ids.length ? item.evidence_ids.join(", ") : "未标注"}
              </div>
            </div>
          ))}
        </section>
      )}
      {questionInfo.length > 0 && (
        <section className="space-y-2">
          <div className="section-label">问题中已知信息 · {questionInfo.length}</div>
          {questionInfo.map((item, index) => (
            <div key={`${item.slot}-${index}`} className="border-b py-3 last:border-0">
              <div className="text-xs font-medium">{item.slot}</div>
              <p className="mt-2 break-words text-xs leading-5 text-muted-foreground">{item.value}</p>
              <div className="mt-2 text-[11px] text-muted-foreground">来源：用户问题</div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function Workspace({ username, onLogout }: { username: string; onLogout: () => Promise<void> }) {
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [selectedDomain, setSelectedDomain] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingDomains, setLoadingDomains] = useState(true);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingAnswer, setLoadingAnswer] = useState(false);
  const [error, setError] = useState("");
  const [logoutError, setLogoutError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [inspectedResponse, setInspectedResponse] = useState<ChatResponse | null>(null);
  const initialized = useRef(false);
  const activeSession = sessions.find((session) => session.session_id === activeSessionId);

  const activeResponse = useMemo(() => {
    const latest = [...messages].reverse().find((message) => message.meta);
    return latest?.meta || null;
  }, [messages]);

  async function loadDomains() {
    setLoadingDomains(true);
    setError("");
    try {
      const result = await fetchDomains();
      setDomains(result.domains);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法加载 domain。");
    } finally {
      setLoadingDomains(false);
    }
  }

  async function reloadDomains(selectDomain?: string) {
    const result = await fetchDomains();
    setDomains(result.domains);
    if (selectDomain) setSelectedDomain(selectDomain);
  }

  async function handleDomainRemoved(domain: string) {
    await reloadDomains();
    if (selectedDomain === domain) setSelectedDomain("");
  }

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void loadDomains();
    void initializeSessions();
  }, []);

  async function refreshSessions() {
    const result = await fetchSessions();
    setSessions(result.sessions);
    return result.sessions;
  }

  async function initializeSessions() {
    setLoadingSessions(true);
    try {
      const availableSessions = await refreshSessions();
      if (availableSessions.length > 0) {
        await selectSession(availableSessions[0].session_id);
        return;
      }
      await createAndSelectSession();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法加载会话。");
    } finally {
      setLoadingSessions(false);
    }
  }

  async function loadSession(sessionId: string) {
    try {
      const result = await fetchSession(sessionId);
      const restored: Message[] = [];
      for (const turn of result.turns) {
        restored.push({
          id: `user-${turn.turn_index}`,
          role: "user",
          content: turn.user_query
        });
        restored.push({
          id: `assistant-${turn.turn_index}`,
          role: "assistant",
          content: turn.assistant_answer
        });
      }
      setMessages(restored);
      setSelectedDomain(result.active_domain || "");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法读取会话历史。");
    }
  }

  async function selectSession(sessionId: string) {
    if (!sessionId || sessionId === activeSessionId || loadingAnswer) return;
    setActiveSessionId(sessionId);
    setMessages([]);
    setQuery("");
    setError("");
    setSidebarOpen(false);
    setInspectedResponse(null);
    await loadSession(sessionId);
  }

  async function createAndSelectSession() {
    if (loadingAnswer) return;
    try {
      const created = await createSession();
      setActiveSessionId(created.session_id);
      setMessages([]);
      setQuery("");
      setError("");
      setSelectedDomain("");
      setSidebarOpen(false);
      setInspectedResponse(null);
      await refreshSessions();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法创建会话。");
    }
  }

  async function renameActiveSession(session: SessionInfo) {
    if (loadingAnswer) return;
    const title = window.prompt("输入新的会话标题：", session.title || "");
    const cleanTitle = title?.trim();
    if (!cleanTitle || cleanTitle === session.title) return;
    try {
      await renameSession(session.session_id, cleanTitle);
      await refreshSessions();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法重命名会话。");
    }
  }

  async function removeSession(session: SessionInfo) {
    if (loadingAnswer) return;
    const title = session.title || "该会话";
    if (!window.confirm(`确定删除“${title}”吗？其历史、事实和证据记忆都会被删除。`)) return;
    try {
      await deleteSession(session.session_id);
      const remaining = await refreshSessions();
      if (session.session_id === activeSessionId) {
        if (remaining.length > 0) {
          await selectSession(remaining[0].session_id);
        } else {
          await createAndSelectSession();
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法删除会话。");
    }
  }

  async function submit() {
    const trimmed = query.trim();
    if (!trimmed || loadingAnswer || !activeSessionId) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);
    setQuery("");
    setLoadingAnswer(true);
    setInspectedResponse(null);
    setError("");
    try {
      const response = await askMoneyAgent({
        query: trimmed,
        domain: selectedDomain,
        session_id: activeSessionId,
        preferred_doc_ids: []
      });
      setMessages((current) => [
        ...current,
        {
          id: response.qid || crypto.randomUUID(),
          role: "assistant",
          content: response.answer || "Agent 没有返回答案。",
          meta: response
        }
      ]);
      await refreshSessions();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "请求失败。";
      setError(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "这轮问答没有完成。",
          error: message
        }
      ]);
    } finally {
      setLoadingAnswer(false);
    }
  }

  async function handleLogout() {
    try {
      setLogoutError("");
      await onLogout();
    } catch (caught) {
      setLogoutError(caught instanceof Error ? caught.message : "退出失败，请重试");
    }
  }

  return (
    <div className="workspace-shell h-dvh overflow-hidden text-foreground">
      {(sidebarOpen || evidenceOpen) && (
        <button type="button" className="fixed inset-0 z-30 bg-foreground/35 lg:hidden"
          onClick={() => { setSidebarOpen(false); setEvidenceOpen(false); }}
          aria-label="关闭侧边面板" />
      )}
      <div className="flex h-full min-h-0">
        <Sidebar
          username={username}
          onLogout={() => void handleLogout()}
          domains={domains}
          selectedDomain={selectedDomain}
          onDomainChange={setSelectedDomain}
          loading={loadingDomains}
          onReload={() => void loadDomains()}
          mobileOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          sessions={sessions}
          activeSessionId={activeSessionId}
          sessionsLoading={loadingSessions}
          disabled={loadingAnswer}
          onSelectSession={(sessionId) => void selectSession(sessionId)}
          onCreateSession={() => void createAndSelectSession()}
          onRenameSession={(session) => void renameActiveSession(session)}
          onDeleteSession={(session) => void removeSession(session)}
          onDomainsChanged={reloadDomains}
          onDomainRemoved={handleDomainRemoved}
        />
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <header className="workspace-header shrink-0 px-4 sm:px-8">
            <div className="flex h-[76px] items-center gap-3">
              <button type="button" onClick={() => setSidebarOpen(true)} className="icon-button lg:hidden"
                aria-label="打开会话列表" title="打开会话列表"><Menu className="h-5 w-5" /></button>
              <div className="min-w-0 flex-1">
                <div className="hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex">
                  <span className="header-path">研究工作台</span>
                  <ChevronRight className="h-3 w-3" />
                  <span>对话</span>
                </div>
                <h1 className="mt-0.5 truncate text-sm font-semibold sm:text-[16px]" title={activeSession?.title || "新研究"}>
                  {activeSession?.title || "新研究"}
                </h1>
              </div>
              <div className="flex shrink-0 items-center gap-1 sm:gap-2">
                <div className="relative">
                  <label htmlFor="domain" className="sr-only">检索知识域</label>
                  <select id="domain" value={selectedDomain} onChange={(event) => setSelectedDomain(event.target.value)}
                    className="domain-select h-9 max-w-[110px] appearance-none truncate rounded-md border pl-3 pr-7 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring sm:max-w-[168px] sm:text-sm">
                    <option value="">全部领域</option>
                    {domains.filter((domain) => domain.page_index_exists).map((domain) => (
                      <option key={domain.domain} value={domain.domain}>{domainLabel(domain)}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2 top-2.5 h-4 w-4 text-muted-foreground" />
                </div>
                <button type="button" onClick={() => void createAndSelectSession()} disabled={loadingAnswer}
                  className="icon-button hidden sm:inline-flex" aria-label="新建会话" title="新建会话"><Plus className="h-4 w-4" /></button>
                <button type="button" onClick={() => setEvidenceOpen((value) => !value)}
                  className={cn("inspector-toggle inline-flex h-9 items-center justify-center gap-2 rounded-md px-2.5 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", evidenceOpen && "inspector-toggle-active")} aria-label={evidenceOpen ? "关闭证据" : "打开证据"}
                  title={evidenceOpen ? "关闭证据" : "打开证据"} aria-expanded={evidenceOpen}>
                  <FileText className="h-4 w-4" /><span className="hidden sm:inline">证据</span>
                </button>
              </div>
            </div>
            {(error || logoutError) && (
              <div role="alert" className="mb-3 flex items-start gap-2 rounded-md border border-destructive/20 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{error || logoutError}
              </div>
            )}
          </header>

          <section className="research-scroll min-h-0 flex-1 overflow-y-auto px-4 sm:px-8">
            {messages.length === 0 ? (
              <div className="mx-auto flex min-h-full max-w-[780px] flex-col justify-center py-8 sm:py-12">
                <div className="empty-lead">
                  <div className="flex items-center gap-3 text-xs font-semibold text-primary">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    RESEARCH DESK
                  </div>
                  <h2 className="mt-6 text-[28px] font-semibold leading-tight sm:text-[32px]">
                    今天想研究什么？
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">
                    提出问题，沿着文档线索继续深入。
                  </p>
                  <div className="mt-6 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
                    <span><strong className="mr-1 font-mono text-foreground">{domains.filter((domain) => domain.page_index_exists).length}</strong>个可用领域</span>
                    <span><strong className="mr-1 font-mono text-foreground">{formatNumber(domains.reduce((total, domain) => total + (domain.page_index_exists ? domain.doc_count : 0), 0))}</strong>份已索引文档</span>
                  </div>
                </div>
                <div className="mt-9 border-t border-foreground/15 pt-4 sm:mt-12">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="section-label">探索方向</h3>
                    <span className="font-mono text-[11px] text-muted-foreground">01 — 03</span>
                  </div>
                  <div className="divide-y">
                    {EXAMPLE_QUESTIONS.map(({ label, question }, index) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => setQuery(question)}
                        className="example-row group flex w-full items-start gap-3 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:items-center sm:gap-5"
                      >
                        <span className="w-6 shrink-0 pt-0.5 font-mono text-xs text-muted-foreground">{String(index + 1).padStart(2, "0")}</span>
                        <span className="min-w-0 flex-1 text-[13px] leading-6 sm:text-sm">{question}</span>
                        <span className="hidden shrink-0 text-[11px] text-muted-foreground md:inline">{label}</span>
                        <ArrowUpRight className="mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-primary" />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="mx-auto max-w-[780px] pb-8 pt-4">
                <div className="mb-2 flex items-center justify-between border-b pb-3 text-[11px] text-muted-foreground">
                  <span className="font-semibold text-foreground">研究记录</span>
                  <span className="font-mono">{Math.ceil(messages.length / 2).toString().padStart(2, "0")} 个问题</span>
                </div>
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} onShowEvidence={(response) => {
                    setInspectedResponse(response);
                    setEvidenceOpen(true);
                  }} />
                ))}
                {loadingAnswer && (
                  <article role="status" className="flex gap-4 py-6">
                    <div className="turn-index turn-index-agent shrink-0">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        正在检索并整理答案
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="h-3 w-3/4 animate-pulse rounded bg-muted" />
                        <div className="h-3 w-1/2 animate-pulse rounded bg-muted" />
                      </div>
                    </div>
                  </article>
                )}
              </div>
            )}
          </section>
          <Composer value={query} onChange={setQuery} onSubmit={submit} loading={loadingAnswer} />
        </main>
        <EvidencePanel activeResponse={inspectedResponse || activeResponse} loading={loadingAnswer && !inspectedResponse}
          open={evidenceOpen} onClose={() => setEvidenceOpen(false)} />
      </div>
    </div>
  );
}

function LoginView({ onLogin, initialError }: {
  onLogin: (username: string, password: string) => Promise<void>;
  initialError: string;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await onLogin(username.trim(), password);
      setPassword("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登录失败，请稍后重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-screen min-h-dvh">
      <header className="flex h-[76px] items-center border-b px-5 sm:px-10">
        <span className="auth-brand-mark" aria-hidden="true">M<span>.</span></span>
        <span className="ml-3 text-sm font-semibold">MoneyAgent</span>
        <span className="ml-auto text-xs text-muted-foreground">研究工作台</span>
      </header>
      <main className="mx-auto flex w-full max-w-[420px] flex-col px-6 pb-12 pt-[min(15vh,110px)] sm:px-4">
        <div className="mb-7 border-l-[3px] border-[#be6546] pl-4">
          <div className="text-xs font-medium text-primary">RESEARCH WORKSPACE</div>
          <h1 className="mt-3 text-2xl font-semibold">欢迎回来</h1>
          <p className="mt-2 text-sm text-muted-foreground">登录后继续你的研究。</p>
        </div>
        <form onSubmit={(event) => void submit(event)} className="space-y-5">
          <div>
            <label htmlFor="auth-username" className="mb-2 block text-sm font-medium">账号</label>
            <input id="auth-username" name="username" autoComplete="username" required minLength={3}
              value={username} onChange={(event) => setUsername(event.target.value)}
              disabled={submitting} className="field h-11" />
          </div>
          <div>
            <label htmlFor="auth-password" className="mb-2 block text-sm font-medium">密码</label>
            <input id="auth-password" name="password" type="password" autoComplete="current-password" required
              value={password} onChange={(event) => setPassword(event.target.value)}
              disabled={submitting} className="field h-11" />
          </div>
          {(error || initialError) && (
            <p role="alert" className="flex gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{error || initialError}
            </p>
          )}
          <button type="submit" disabled={submitting} className="send-button h-11 w-full">
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {submitting ? "正在登录" : "登录"}
          </button>
        </form>
        <p className="mt-6 text-xs text-muted-foreground">账号由管理员邀请开通。</p>
      </main>
    </div>
  );
}

export function App() {
  const [username, setUsername] = useState("");
  const [checking, setChecking] = useState(true);
  const [initialError, setInitialError] = useState("");

  useEffect(() => {
    let active = true;
    const onUnauthorized = () => {
      if (active) {
        setUsername("");
        setInitialError("登录已过期，请重新登录");
      }
    };
    window.addEventListener("moneyagent:unauthorized", onUnauthorized);
    void fetchCurrentUser()
      .then((user) => { if (active) { setUsername(user.username); setInitialError(""); } })
      .catch((caught) => {
        if (active && !(caught instanceof ApiError && caught.status === 401)) {
          setInitialError("无法连接服务，请确认后端已启动");
        }
      })
      .finally(() => { if (active) setChecking(false); });
    return () => { active = false; window.removeEventListener("moneyagent:unauthorized", onUnauthorized); };
  }, []);

  if (checking) {
    return <div role="status" className="flex min-h-dvh items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />正在验证登录
    </div>;
  }
  if (!username) {
    return <LoginView initialError={initialError} onLogin={async (name, password) => {
      const user = await login(name, password);
      setInitialError("");
      setUsername(user.username);
    }} />;
  }
  return <Workspace username={username} onLogout={async () => {
    await logout();
    setUsername("");
  }} />;
}
