import {
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Database,
  FileUp,
  FileText,
  FolderPlus,
  Loader2,
  MessageSquareText,
  Pencil,
  Plus,
  RefreshCcw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Trash2,
  User
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  askMoneyAgent,
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
  fetchDomains,
  fetchSession,
  fetchSessions,
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
  "请比较平安e生保和太保团体百万医疗的免赔额规则。",
  "东方甄选相关研报中，GMV 和自营产品表现有什么变化？",
  "请根据金融合同说明债券应计利息的计算逻辑。"
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

function StatusBadge({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        ok
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-destructive/20 bg-destructive/10 text-destructive"
      )}
    >
      {ok ? <CheckCircle2 className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
      {children}
    </span>
  );
}

function Sidebar({
  domains,
  selectedDomain,
  onDomainChange,
  loading,
  onReload,
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
  domains: DomainInfo[];
  selectedDomain: string;
  onDomainChange: (domain: string) => void;
  loading: boolean;
  onReload: () => void;
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
    <aside className="border-b bg-card lg:h-screen lg:min-h-0 lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="flex min-h-full flex-col gap-5 p-4 lg:p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold">
              <ShieldCheck className="h-4 w-4" />
              MoneyAgent
            </div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">金融文档问答工作台</p>
          </div>
          <button
            type="button"
            onClick={onReload}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border bg-background text-muted-foreground hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="刷新索引状态"
          >
            <RefreshCcw className={cn("h-4 w-4", loading && "animate-spin")} />
          </button>
        </div>

        <section className="shrink-0 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium">对话</h2>
            <button
              type="button"
              onClick={onCreateSession}
              disabled={disabled}
              className="inline-flex h-7 items-center gap-1 rounded-md border bg-background px-2 text-xs font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
            >
              <Plus className="h-3.5 w-3.5" />
              新建
            </button>
          </div>
          <div className="max-h-64 space-y-1 overflow-y-auto pr-1 lg:max-h-[34vh]">
            {sessionsLoading && sessions.length === 0 ? (
              Array.from({ length: 3 }).map((_, index) => (
                <div key={index} className="h-14 animate-pulse rounded-md border bg-muted/50" />
              ))
            ) : sessions.length === 0 ? (
              <p className="rounded-md border border-dashed p-3 text-xs leading-5 text-muted-foreground">暂无会话，点击“新建”开始。</p>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={cn(
                    "group flex items-center gap-1 rounded-md border p-1 transition-colors hover:bg-muted/50",
                    activeSessionId === session.session_id && "border-foreground bg-muted"
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelectSession(session.session_id)}
                    disabled={disabled}
                    className="min-w-0 flex-1 rounded-sm px-2 py-1.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed"
                  >
                    <div className="truncate text-xs font-medium">{session.title || "新会话"}</div>
                    <div className="mt-1 text-[11px] text-muted-foreground">{session.turn_count} 轮对话</div>
                  </button>
                  <div className="flex shrink-0 opacity-100 lg:opacity-0 lg:group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => onRenameSession(session)}
                      disabled={disabled}
                      className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                      aria-label={`重命名 ${session.title || "会话"}`}
                      title="重命名"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteSession(session)}
                      disabled={disabled}
                      className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-background hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
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

        <section className="space-y-2">
          <label htmlFor="domain" className="text-sm font-medium">
            知识域
          </label>
          <div className="relative">
            <select
              id="domain"
              value={selectedDomain}
              onChange={(event) => onDomainChange(event.target.value)}
              className="h-9 w-full appearance-none rounded-md border bg-background px-3 pr-9 text-sm outline-none hover:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value="">自动识别</option>
              {domains.filter((domain) => domain.page_index_exists).map((domain) => (
                <option key={domain.domain} value={domain.domain}>
                  {domainLabel(domain)}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          </div>
          <p className="text-xs leading-5 text-muted-foreground">自动识别会先做跨域目录检索，再选择对应 page index。</p>
        </section>

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium">索引状态</h2>
            <button
              type="button"
              onClick={() => setShowManager((value) => !value)}
              aria-expanded={showManager}
              className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <FolderPlus className="h-3.5 w-3.5" />
              管理领域
            </button>
          </div>
          {showManager && (
            <div className="space-y-3 border-y py-3 text-xs">
              <form onSubmit={(event) => void create(event)} className="space-y-2">
                <div className="text-sm font-medium">创建领域</div>
                <label htmlFor="domain-name" className="block text-muted-foreground">领域名称</label>
                <input id="domain-name" value={domainName} onChange={(event) => setDomainName(event.target.value)}
                  maxLength={60} required placeholder="例如：新能源行业"
                  className="h-9 w-full rounded-md border bg-background px-2 outline-none focus-visible:ring-2 focus-visible:ring-ring" />
                <label htmlFor="domain-key" className="block text-muted-foreground">领域标识（英文）</label>
                <input id="domain-key" value={domainKey} onChange={(event) => setDomainKey(event.target.value)}
                  required pattern="[a-z][a-z0-9_-]{2,39}" placeholder="例如：new_energy"
                  className="h-9 w-full rounded-md border bg-background px-2 outline-none focus-visible:ring-2 focus-visible:ring-ring" />
                <label htmlFor="split-mode" className="block text-muted-foreground">切分方式</label>
                <select id="split-mode" value={splitMode} onChange={(event) => setSplitMode(event.target.value)}
                  className="h-9 w-full rounded-md border bg-background px-2 outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  <option value="heading">Markdown 标题</option>
                  <option value="heading_plus_numbered">标题与编号条款</option>
                  <option value="block">段落</option>
                </select>
                <button type="submit" disabled={!!domainBusy || disabled}
                  className="h-8 rounded-md border px-3 font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
                  创建
                </button>
              </form>
              {customDomains.length > 0 && (
                <>
                  <div className="border-t pt-3 text-sm font-medium">上传与构建</div>
                  <label htmlFor="managed-domain" className="block text-muted-foreground">目标领域</label>
                  <select id="managed-domain" value={targetDomain} onChange={(event) => setManagedDomain(event.target.value)}
                    className="h-9 w-full rounded-md border bg-background px-2 outline-none focus-visible:ring-2 focus-visible:ring-ring">
                    {customDomains.map((domain) => (
                      <option key={domain.domain} value={domain.domain}>{domainLabel(domain)}</option>
                    ))}
                  </select>
                  <form onSubmit={(event) => void upload(event)} className="space-y-2">
                    <label htmlFor="domain-files" className="block text-muted-foreground">UTF-8 Markdown 或文本文件（每个不超过 8 MB）</label>
                    <input id="domain-files" name="domain-files" type="file" accept=".md,.txt" multiple required
                      className="block w-full text-xs file:mr-2 file:rounded-md file:border file:bg-background file:px-2 file:py-1.5 file:text-xs" />
                    <div className="flex items-center gap-2">
                      <button type="submit" disabled={!!domainBusy || disabled}
                        className="h-8 rounded-md border px-3 font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
                        上传
                      </button>
                      <button type="button" onClick={() => void build()}
                        disabled={!!domainBusy || disabled || !(customDomains.find((domain) => domain.domain === targetDomain)?.upload_count)}
                        className="h-8 rounded-md bg-primary px-3 font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
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
                    className="h-8 rounded-md border border-destructive/30 px-3 text-destructive hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
                    删除领域
                  </button>
                </>
              )}
              {domainBusy && <div role="status" className="text-muted-foreground">{domainBusy}，请稍候…</div>}
              {domainError && <div role="alert" className="text-destructive">{domainError}</div>}
              {domainNotice && <div role="status" className="text-muted-foreground">{domainNotice}</div>}
            </div>
          )}
          <div className="space-y-2">
            {loading && domains.length === 0
              ? Array.from({ length: 5 }).map((_, index) => (
                  <div key={index} className="h-16 animate-pulse rounded-lg border bg-muted/50" />
                ))
              : domains.map((domain) => (
                  <button
                    key={domain.domain}
                    type="button"
                    onClick={() => onDomainChange(domain.page_index_exists ? domain.domain : selectedDomain)}
                    disabled={!domain.page_index_exists}
                    className={cn(
                      "w-full rounded-lg border p-3 text-left text-sm transition hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60",
                      selectedDomain === domain.domain && "border-foreground bg-muted"
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-medium">{domainLabel(domain)}</span>
                      <StatusBadge ok={domain.catalog_exists && domain.page_index_exists}>
                        {domain.status === "needs_build" ? "待重建" : domain.page_index_exists ? "ready" : "待构建"}
                      </StatusBadge>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <span>{formatNumber(domain.doc_count)} 文档</span>
                      <span>{domain.page_index_exists ? `${formatNumber(domain.page_count)} 页面` : `${formatNumber(domain.upload_count)} 待构建文件`}</span>
                    </div>
                  </button>
                ))}
          </div>
        </section>

        <section className="mt-auto rounded-lg border bg-muted/40 p-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Server className="h-4 w-4" />
            API
          </div>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">后端默认代理到 127.0.0.1:8000，前端开发端口为 5173。</p>
        </section>
      </div>
    </aside>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={cn("flex gap-3", isAssistant ? "items-start" : "items-start justify-end")}>
      {isAssistant && (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border bg-muted">
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div className={cn("max-w-[860px] rounded-lg border px-4 py-3", isAssistant ? "bg-card" : "bg-foreground text-background")}>
        <div className="whitespace-pre-wrap text-sm leading-6">{message.content}</div>
        {message.error && <p className="mt-2 text-xs text-destructive">{message.error}</p>}
        {message.meta && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span>domain: {message.meta.domain || "auto"}</span>
            <span>docs: {message.meta.doc_ids.join(", ") || "-"}</span>
            <span>tokens: {message.meta.token_usage.total_tokens}</span>
          </div>
        )}
      </div>
      {!isAssistant && (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border bg-card">
          <User className="h-4 w-4" />
        </div>
      )}
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
    <form onSubmit={submit} className="border-t bg-background p-4">
      <div className="rounded-lg border bg-card p-2">
        <label htmlFor="query" className="sr-only">
          输入金融问题
        </label>
        <textarea
          id="query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          rows={3}
          placeholder="输入金融文档问题，例如：请比较两个保险产品的免赔额规则"
          className="min-h-20 w-full resize-none bg-transparent px-2 py-2 text-sm leading-6 outline-none placeholder:text-muted-foreground"
          disabled={loading}
        />
        <div className="flex flex-col gap-2 border-t pt-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((question) => (
              <button
                key={question}
                type="button"
                onClick={() => onChange(question)}
                className="rounded-md border px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {question.length > 18 ? `${question.slice(0, 18)}...` : question}
              </button>
            ))}
          </div>
          <button
            type="submit"
            disabled={loading || !value.trim()}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            发送
          </button>
        </div>
      </div>
    </form>
  );
}

function EvidencePanel({ activeResponse, loading }: { activeResponse: ChatResponse | null; loading: boolean }) {
  const evidence = activeResponse?.evidence || [];
  const facts = activeResponse?.facts || [];
  const questionInfo = activeResponse?.question_info || [];
  return (
    <aside className="max-h-[65vh] min-h-0 border-t bg-card lg:h-screen lg:max-h-none lg:w-96 lg:border-l lg:border-t-0">
      <div className="flex h-full min-h-0 flex-col">
        <div className="border-b p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <FileText className="h-4 w-4" />
            证据与调试
          </div>
          <p className="mt-1 text-xs text-muted-foreground">展示 Agent 返回的证据、文档命中和 token 使用。</p>
        </div>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {loading && (
            <div className="space-y-2">
              <div className="h-20 animate-pulse rounded-lg border bg-muted/50" />
              <div className="h-20 animate-pulse rounded-lg border bg-muted/50" />
            </div>
          )}
          {!loading && !activeResponse && (
            <div className="rounded-lg border border-dashed p-4 text-sm">
              <div className="font-medium">暂无证据</div>
              <p className="mt-1 leading-5 text-muted-foreground">发送一个问题后，这里会显示命中文档、证据片段和运行信息。</p>
            </div>
          )}
          {activeResponse && (
            <>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Domain</div>
                  <div className="mt-1 truncate font-medium">{activeResponse.domain || "auto"}</div>
                </div>
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Token</div>
                  <div className="mt-1 font-medium">{formatNumber(activeResponse.token_usage.total_tokens)}</div>
                </div>
              </div>
              <EvidenceList evidence={evidence} facts={facts} questionInfo={questionInfo} />
              <div className="rounded-lg border p-3 text-xs">
                <div className="mb-2 flex items-center gap-2 font-medium">
                  <Database className="h-3.5 w-3.5" />
                  Debug files
                </div>
                <div className="space-y-1 text-muted-foreground">
                  {Object.entries(activeResponse.debug_files || {}).map(([key, value]) => (
                    <div key={key} className="truncate">
                      {key}: {value}
                    </div>
                  ))}
                </div>
              </div>
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
      <div className="rounded-lg border border-dashed p-4 text-sm">
        <div className="font-medium">没有返回证据片段</div>
        <p className="mt-1 text-muted-foreground">可能是问答失败、证据为空，或当前输出没有包含 evidence summary。</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {evidence.length > 0 && (
        <section className="space-y-2">
          <div className="text-sm font-medium">文档证据</div>
      {evidence.map((item, index) => (
        <div key={`${item.doc_id}-${item.page_id}-${index}`} className="rounded-lg border p-3">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>文档 {item.doc_id || "未知"}</span>
            <span>页面 {item.page_id || item.evidence_id || "未知"}</span>
          </div>
          {item.heading && <div className="mt-2 text-xs font-medium">{item.heading}</div>}
          <p className="mt-2 line-clamp-6 whitespace-pre-wrap text-xs leading-5 text-muted-foreground">{item.quote || "无原文摘录"}</p>
        </div>
      ))}
        </section>
      )}
      {facts.length > 0 && (
        <section className="space-y-2">
          <div className="text-sm font-medium">回答要点</div>
          {facts.map((item, index) => (
            <div key={`${item.slot}-${index}`} className="rounded-lg border p-3">
              <div className="text-xs font-medium">{item.slot}</div>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">{item.value}</p>
              <div className="mt-2 text-[11px] text-muted-foreground">
                来源：{item.evidence_ids.length ? item.evidence_ids.join(", ") : "未标注"}
              </div>
            </div>
          ))}
        </section>
      )}
      {questionInfo.length > 0 && (
        <section className="space-y-2">
          <div className="text-sm font-medium">问题中已知信息</div>
          {questionInfo.map((item, index) => (
            <div key={`${item.slot}-${index}`} className="rounded-lg border border-dashed p-3">
              <div className="text-xs font-medium">{item.slot}</div>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">{item.value}</p>
              <div className="mt-2 text-[11px] text-muted-foreground">来源：用户问题</div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

export function App() {
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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden lg:flex-row">
        <Sidebar
          domains={domains}
          selectedDomain={selectedDomain}
          onDomainChange={setSelectedDomain}
          loading={loadingDomains}
          onReload={() => void loadDomains()}
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
        <main className="flex min-h-[720px] flex-1 flex-col lg:min-h-0">
          <header className="border-b bg-card px-4 py-4 lg:px-6">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2 text-xl font-semibold tracking-tight">
                  <MessageSquareText className="h-5 w-5" />
                  {activeSession?.title || "金融问答"}
                </div>
                <p className="mt-1 text-sm text-muted-foreground">基于金融合同、财报、保险条款、监管文件与研报的检索增强问答。</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusBadge ok={!error}>API {error ? "error" : "ready"}</StatusBadge>
                <StatusBadge ok={domains.length > 0}>Index {domains.length || 0}</StatusBadge>
                <button
                  type="button"
                  onClick={() => void createAndSelectSession()}
                  disabled={loadingAnswer}
                  className="inline-flex h-7 items-center gap-1 rounded-md border bg-background px-2 text-xs font-medium hover:bg-muted disabled:opacity-50"
                >
                  <Plus className="h-3.5 w-3.5" />
                  新会话
                </button>
              </div>
            </div>
            {error && (
              <div className="mt-3 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}
          </header>

          <section className="min-h-0 flex-1 overflow-y-auto p-4 lg:p-6">
            {messages.length === 0 ? (
              <div className="mx-auto flex min-h-[420px] max-w-3xl flex-col justify-center">
                <div className="rounded-xl border bg-card p-5">
                  <div className="flex items-center gap-2 text-base font-semibold">
                    <Search className="h-4 w-4" />
                    开始一次金融文档问答
                  </div>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                    选择一个知识域，或保持自动识别。提问后，右侧会展示命中文档、证据片段和 token 使用情况。
                  </p>
                  <div className="mt-4 grid gap-2">
                    {EXAMPLE_QUESTIONS.map((question) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => setQuery(question)}
                        className="rounded-lg border px-3 py-2 text-left text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {loadingAnswer && (
                  <article className="flex gap-3">
                    <div className="mt-1 flex h-8 w-8 items-center justify-center rounded-md border bg-muted">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="w-full max-w-[860px] rounded-lg border bg-card px-4 py-3">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        MoneyAgent 正在检索、规划并生成答案
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
        <EvidencePanel activeResponse={activeResponse} loading={loadingAnswer} />
      </div>
    </div>
  );
}
