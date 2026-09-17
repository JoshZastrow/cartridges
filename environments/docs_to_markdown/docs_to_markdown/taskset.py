from pathlib import Path

import verifiers.v1 as vf

from cartridges.datagen import LOADERS, convert_file

# NOTE: reaches up out of this installable package into the repo's local data/
# dir. Fine for local dev; revisit with a configurable data root before this
# environment is ever installed/published standalone.
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


class DocsToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "docs"

    @vf.tool
    async def list_files(self, source_dir: str) -> list[str]:
        """List convertible files in a source directory (relative to data/)."""
        root = DATA_ROOT / source_dir
        return sorted(
            p.name for p in root.iterdir() if p.suffix.lower() in LOADERS
        )

    @vf.tool
    async def convert(self, source_dir: str, filename: str) -> str:
        """Convert one file under source_dir to markdown and return its content."""
        return convert_file(DATA_ROOT / source_dir / filename)

    @vf.tool
    async def write_markdown(self, output_path: str, content: str) -> str:
        """Write markdown content to output_path (relative to data/)."""
        path = DATA_ROOT / output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return output_path


class DocsToMarkdownData(vf.TaskData):
    source_dir: str  # relative to data/, e.g. "journals"
    output_path: str  # relative to data/, e.g. "journals.md"


class DocsToMarkdownTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class DocsToMarkdownTask(vf.Task[DocsToMarkdownData, vf.State, DocsToMarkdownTaskConfig]):
    @classmethod
    def toolsets(cls, config: DocsToMarkdownTaskConfig) -> list[vf.Toolset]:
        return [DocsToolset(config.tools)]

    @vf.reward
    async def output_written(self, trace: vf.Trace) -> float:
        path = DATA_ROOT / self.data.output_path
        return float(path.exists() and path.stat().st_size > 0)


class DocsToMarkdownConfig(vf.TasksetConfig):
    task: DocsToMarkdownTaskConfig = DocsToMarkdownTaskConfig()


class DocsToMarkdownTaskset(vf.Taskset[DocsToMarkdownTask, DocsToMarkdownConfig]):
    def load(self) -> list[DocsToMarkdownTask]:
        subdirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
        return [
            DocsToMarkdownTask(
                DocsToMarkdownData(
                    idx=i,
                    prompt=(
                        f"Convert every document under data/{subdir.name} into a "
                        f"single markdown file at data/{subdir.name}.md using the "
                        f"docs tools."
                    ),
                    source_dir=subdir.name,
                    output_path=f"{subdir.name}.md",
                ),
                self.config.task,
            )
            for i, subdir in enumerate(subdirs)
        ]


__all__ = ["DocsToMarkdownTaskset"]
