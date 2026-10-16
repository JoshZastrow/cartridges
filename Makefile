.PHONY: setup notebook prepare-data chat-up chat-down chat-test chat-status chat-clean

setup:
	uv sync

notebook:
	uv run marimo edit notebooks/

prepare-data:
	set -a && . ./.env.local && set +a && uv run cartridges prepare_data "$(INPUT)" "$(OUTPUT)" --model "$(MODEL)"

# LibreChat on the journal Teacher (docs/plans/librechat.md).
MODEL ?= mock-echo
TURN_TIMEOUT_MS ?= 1500000

chat-up:        ## gateway (8008) + Mongo + LibreChat (http://localhost:3080)
	./librechat/start.sh

chat-down:      ## stop all three and this app's idle Modal GPU containers
	./librechat/stop.sh

chat-test:      ## 3-turn Playwright chat; starts the stack if needed. MODEL=journal-teacher-27b
	LIBRECHAT_MODEL=$(MODEL) PW_TURN_TIMEOUT_MS=$(TURN_TIMEOUT_MS) npx playwright test e2e/librechat.spec.ts

chat-status:    ## what is listening locally and running on Modal
	@lsof -nP -iTCP:3080 -iTCP:8008 -iTCP:27017 -sTCP:LISTEN | awk 'NR>1{print $$1, $$9}' || true
	@uv run modal container list

chat-clean: chat-down   ## also delete Mongo data (users, chats), logs and Playwright output
	rm -rf librechat/.mongo-data librechat/logs test-results
