"""
Display Screens - Messages screen mixin.

Renders messages, questions, visitors screens and handles scroll/expand interactions.
"""

import sys
import time


# --- Q&A screen layout -------------------------------------------------------
# The 240x240 LCD is the ONLY place Lumen's questions are ever shortened: the
# stored text, the lumen_qa MCP tool and the /qa REST endpoint all carry the
# question in full. Naming the geometry keeps the three Q&A views (list /
# expanded / full) from drifting apart again — each previously guessed its own
# character slice, and two of the three guessed wrong.
QA_LINE_H = 12          # px per text line at fonts['small'] (11px + leading)
QA_STATUS_BAR_Y = 218   # y of the status bar; content must stop above it
QA_Q_HEADER_X = 30      # full-view question text x (clear of the 'Q:' label at x=12)
QA_Q_HEADER_W = 198     # ...to the panel edge at 234, less 6 pad
QA_Q_HEADER_LINES = 2   # lines the full-view question header may use


class MessagesMixin:
    """Mixin for message-group screens (messages, questions, visitors) and interaction methods."""

    def _render_messages(self):
        """Render message board. List view with type bars + context-aware reading view."""
        try:
            from ..messages import get_recent_messages, MESSAGE_TYPE_USER, MESSAGE_TYPE_OBSERVATION, MESSAGE_TYPE_AGENT, MESSAGE_TYPE_QUESTION

            all_messages = get_recent_messages(50)
            scroll_idx = self._state.message_scroll_index
            expanded_id = self._state.message_expanded_id

            # Cache check — text rendering is expensive (~500ms)
            cache_hash = self._get_messages_cache_hash(all_messages, scroll_idx, expanded_id)
            if self._messages_cache_image is not None and self._messages_cache_hash == cache_hash:
                if hasattr(self._display, '_image'):
                    self._display._image = self._messages_cache_image.copy()
                if hasattr(self._display, '_show'):
                    self._display._show()
                return

            if not hasattr(self._display, '_create_canvas'):
                # Text-only fallback
                msgs = get_recent_messages(6)
                lines = ["MESSAGES", ""]
                for m in msgs:
                    pfx = "\u25b8 " if m.msg_type == MESSAGE_TYPE_OBSERVATION else ("\u25c6 " if m.msg_type == MESSAGE_TYPE_AGENT else "\u25cf ")
                    lines.append(pfx + m.text[:22])
                self._display.render_text("\n".join(lines), (10, 10))
                return

            image, draw = self._display._create_canvas((0, 0, 0))

            # ── Palette ──────────────────────────────────────────────────────
            # Type accent colors: bar, label (full saturation for visual anchors)
            C_OBS   = (140,  90, 210)   # soft violet  — Lumen observations
            C_USER  = ( 75, 200,  85)   # lime green   — you
            C_AGENT = (215, 160,  45)   # warm amber   — agents
            C_QA    = ( 55, 185, 225)   # sky cyan     — questions
            # Content text (softer — readable in bulk without overwhelming)
            T_OBS   = (210, 190, 130)   # warm gold
            T_USER  = (185, 230, 185)   # soft lime
            T_AGENT = (228, 200, 160)   # soft cream
            T_QA    = (160, 218, 228)   # soft cyan
            # UI chrome
            SEL_BG  = ( 22,  36,  58)
            SEL_BR  = ( 55,  90, 145)
            MUTED   = ( 88, 105, 125)   # timestamps, hints — clearly secondary
            DIV     = ( 30,  42,  62)   # divider lines
            TITLE   = (100, 200, 240)   # screen title

            def _type_colors(msg_type):
                if msg_type == MESSAGE_TYPE_USER:
                    return C_USER, T_USER
                if msg_type == MESSAGE_TYPE_AGENT:
                    return C_AGENT, T_AGENT
                if msg_type == MESSAGE_TYPE_QUESTION:
                    return C_QA, T_QA
                return C_OBS, T_OBS

            def _dim(color, f):
                return tuple(int(c * f) for c in color)

            def _short_author(msg):
                if msg.msg_type == MESSAGE_TYPE_OBSERVATION:
                    return "lumen"
                if msg.msg_type == MESSAGE_TYPE_USER:
                    return "you"
                if msg.msg_type == MESSAGE_TYPE_QUESTION:
                    return "\u2713 lumen" if getattr(msg, 'answered', False) else "? lumen"
                a = getattr(msg, 'author', None) or 'agent'
                return a[:14]

            # ── Fonts ─────────────────────────────────────────────────────────
            fonts  = self._get_fonts()
            f_titl = fonts['default']   # 14px  — screen title
            f_meta = fonts['small']     # 11px  — author, age, hints
            f_body = fonts['small']     # 11px  — message body

            # ── Layout ────────────────────────────────────────────────────────
            LINE      = 13    # line height  (11px font + 2px leading)
            PAD       = 4     # inner padding top + bottom per row
            GAP       = 3     # gap between rows in list view
            LIST_X    = 13    # text left margin (right of the 3px bar)
            MAX_Y     = 213   # bottom content boundary
            COMPACT_H = PAD + LINE + LINE + PAD   # 30px — author + 1 body line
            SEL_H     = PAD + LINE + LINE + LINE + PAD  # 43px — author + 2 body lines
            STRIP_H   = LINE + 7   # 20px — context strip in reading view

            # ── Title ─────────────────────────────────────────────────────────
            draw.text((10, 5), "messages", fill=TITLE, font=f_titl)

            if not all_messages:
                draw.text((68, 108), "nothing yet",    fill=MUTED, font=f_meta)
                draw.text((72, 122), "be patient",     fill=_dim(MUTED, 0.55), font=f_meta)
            else:
                scroll_idx = max(0, min(scroll_idx, len(all_messages) - 1))
                self._state.message_scroll_index = scroll_idx
                n = len(all_messages)
                has_expanded = self._state.message_expanded_id is not None

                if has_expanded:
                    # ── READING VIEW ─────────────────────────────────────────
                    # Show the selected message in full, with thin context strips
                    # above (previous) and below (next) to preserve orientation.
                    msg = all_messages[scroll_idx]
                    bar_c, text_c = _type_colors(msg.msg_type)
                    author = _short_author(msg)
                    age    = msg.age_str()

                    prev_msg = all_messages[scroll_idx - 1] if scroll_idx > 0      else None
                    next_msg = all_messages[scroll_idx + 1] if scroll_idx + 1 < n  else None

                    y = 22

                    # Previous context strip (dim, just for orientation)
                    if prev_msg:
                        pb, _ = _type_colors(prev_msg.msg_type)
                        draw.rectangle([6, y, 9, y + STRIP_H], fill=_dim(pb, 0.32))
                        draw.rectangle([9, y, 234, y + STRIP_H], fill=(11, 14, 22))
                        draw.text((LIST_X, y + 4),         _short_author(prev_msg), fill=_dim(pb, 0.45),  font=f_meta)
                        draw.text((183,    y + 4),          prev_msg.age_str(),      fill=_dim(MUTED, 0.5), font=f_meta)
                        trunc = prev_msg.text[:36] + "\u2026" if len(prev_msg.text) > 36 else prev_msg.text
                        draw.text((LIST_X, y + 4 + LINE),  trunc,                   fill=_dim(MUTED, 0.4), font=f_meta)
                        y += STRIP_H
                        draw.line([(6, y), (234, y)], fill=DIV)
                        y += 2

                    # Reading header (who/when/position — always visible context)
                    hdr_top = y
                    hdr_h   = LINE + 7
                    draw.rectangle([6,  y, 234, y + hdr_h], fill=(16, 22, 36))
                    draw.rectangle([6,  y,   9, y + hdr_h], fill=bar_c)
                    pos_str = f"{scroll_idx + 1}\u2009/\u2009{n}"
                    draw.text((LIST_X, y + 3), author,  fill=bar_c, font=f_meta)
                    draw.text((105,    y + 3), pos_str, fill=MUTED, font=f_meta)
                    draw.text((183,    y + 3), age,     fill=MUTED, font=f_meta)
                    y += hdr_h
                    draw.line([(6, y), (234, y)], fill=DIV)
                    y += 4

                    # How much room until next strip (or bottom)
                    next_strip_top = MAX_Y - (STRIP_H + 2) if next_msg else MAX_Y
                    text_area_h    = next_strip_top - y
                    max_lines      = max(1, (text_area_h - 2) // LINE)

                    wrapped   = self._wrap_text(msg.text, f_body, 218)
                    ts        = self._state.message_text_scroll
                    max_scroll= max(0, len(wrapped) - max_lines)
                    ts        = min(ts, max_scroll)
                    self._state.message_text_scroll = ts

                    ty = y
                    for line in wrapped[ts: ts + max_lines]:
                        if ty + LINE > next_strip_top - 2:
                            break
                        draw.text((LIST_X, ty), line, fill=text_c, font=f_body)
                        ty += LINE

                    # Scroll arrows — right margin, bracketing the text area
                    if len(wrapped) > max_lines:
                        if ts > 0:
                            draw.text((226, hdr_top + hdr_h + 5), "\u25b2", fill=MUTED, font=f_meta)
                        if ts < max_scroll:
                            draw.text((226, next_strip_top - LINE - 2), "\u25bc", fill=MUTED, font=f_meta)

                    # Next context strip (dim)
                    if next_msg:
                        ny = next_strip_top
                        nb, _ = _type_colors(next_msg.msg_type)
                        draw.line([(6, ny - 2), (234, ny - 2)], fill=DIV)
                        draw.rectangle([6,  ny, 9,   ny + STRIP_H], fill=_dim(nb, 0.32))
                        draw.rectangle([9,  ny, 234, ny + STRIP_H], fill=(11, 14, 22))
                        draw.text((LIST_X, ny + 4),        _short_author(next_msg), fill=_dim(nb, 0.45),   font=f_meta)
                        draw.text((183,    ny + 4),         next_msg.age_str(),      fill=_dim(MUTED, 0.5), font=f_meta)
                        trunc_n = next_msg.text[:36] + "\u2026" if len(next_msg.text) > 36 else next_msg.text
                        draw.text((LIST_X, ny + 4 + LINE), trunc_n,                 fill=_dim(MUTED, 0.4), font=f_meta)

                else:
                    # ── LIST VIEW ────────────────────────────────────────────
                    start_idx = max(0, scroll_idx - 2)
                    end_idx   = min(n, start_idx + 5)
                    if end_idx - start_idx < 5:
                        start_idx = max(0, end_idx - 5)
                    sel_in_vis = scroll_idx - start_idx

                    y = 22
                    for i, msg in enumerate(all_messages[start_idx:end_idx]):
                        if y > MAX_Y - COMPACT_H:
                            break

                        is_sel  = (i == sel_in_vis)
                        bar_c, text_c = _type_colors(msg.msg_type)
                        author = _short_author(msg)
                        age    = msg.age_str()
                        row_h  = SEL_H if is_sel else COMPACT_H

                        # Row background
                        if is_sel:
                            draw.rectangle([6, y, 234, y + row_h], fill=SEL_BG, outline=SEL_BR, width=1)
                        else:
                            draw.rectangle([6, y, 234, y + row_h], fill=(11, 14, 22))

                        # Left-edge type bar (3px) — immediate visual type cue
                        draw.rectangle([6, y, 9, y + row_h], fill=bar_c if is_sel else _dim(bar_c, 0.5))

                        inner = y + PAD

                        # Author + age — always on the same line, age right-aligned
                        label_c = bar_c if is_sel else _dim(bar_c, 0.65)
                        draw.text((LIST_X, inner), author, fill=label_c, font=f_meta)
                        draw.text((183,    inner), age,    fill=MUTED,   font=f_meta)
                        inner += LINE

                        # Text
                        if is_sel:
                            # 2-line wrapped preview — shows enough to decide whether to read
                            for line in self._wrap_text(msg.text, f_body, 215)[:2]:
                                if inner + LINE > y + row_h:
                                    break
                                draw.text((LIST_X, inner), line, fill=text_c, font=f_body)
                                inner += LINE
                        else:
                            # 1 truncated line — dimmer, just for scanning
                            trunc = msg.text[:34] + "\u2026" if len(msg.text) > 34 else msg.text
                            draw.text((LIST_X, inner), trunc, fill=_dim(text_c, 0.55), font=f_body)

                        y += row_h + GAP

                    # Scrollbar (right edge, 3px wide)
                    if n > 5:
                        sb_top, sb_h = 22, 185
                        thumb_h   = max(14, sb_h * 5 // n)
                        thumb_top = sb_top + int((scroll_idx / max(1, n - 1)) * (sb_h - thumb_h))
                        draw.rectangle([234, sb_top, 237, sb_top + sb_h], fill=(16, 20, 30))
                        draw.rectangle([234, thumb_top, 237, thumb_top + thumb_h], fill=(60, 85, 130))

                    # Position counter (bottom-left, very dim)
                    draw.text((10, 216), f"{scroll_idx + 1}\u2009/\u2009{n}", fill=MUTED, font=f_meta)
                    draw.text((80, 216), "\u25c0\u25b6 q&a/visitors  btn:read", fill=MUTED, font=f_meta)

            self._draw_status_bar(draw)

            # Cache
            self._messages_cache_image = image.copy()
            self._messages_cache_hash  = cache_hash
            if hasattr(self._display, '_image'):
                self._display._image = image
            if hasattr(self._display, '_show'):
                self._display._show()

        except Exception as e:
            import traceback
            print(f"[Messages Screen] Error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._display.render_text("MESSAGES\n\nError", (10, 10))

    def _render_questions(self):
        """Render Questions screen - Lumen's questions and answers."""
        # Use the proper Q&A renderer instead of filtered messages
        try:
            self._render_qa_content()
        except Exception as e:
            import traceback
            print(f"[Questions Screen] Error in _render_qa_content: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            # Fallback: show error message
            try:
                self._display.render_text("Q&A\n\nError\nrendering", (10, 10))
            except Exception:
                pass

    def _render_visitors(self):
        """Render Visitors screen - messages from agents and humans."""
        # Note: "user" is the actual msg_type for human messages (not "human")
        self._render_filtered_messages("visitors", ["agent", "user"], include_answers=False)

    def _render_filtered_messages(self, title: str, filter_types: list, include_answers: bool):
        """Render a filtered message screen (visitors). Same design language as messages screen."""
        try:
            from ..messages import get_board, MESSAGE_TYPE_USER, MESSAGE_TYPE_AGENT, MESSAGE_TYPE_QUESTION

            board = get_board()
            board._load()
            all_messages = board._messages
            scroll_idx  = getattr(self._state, 'message_scroll_index', 0)
            expanded_id = getattr(self._state, 'message_expanded_id', None)
            msg_ids     = "|".join(m.message_id for m in all_messages[-15:]) if all_messages else ""
            cache_key   = f"{title}|{msg_ids}|{scroll_idx}|{expanded_id or ''}"
            if self._check_screen_cache(title, cache_key):
                return

            if not hasattr(self._display, '_create_canvas'):
                self._display.render_text(f"{title.upper()}\n\nNo display", (10, 10))
                return

            image, draw = self._display._create_canvas((0, 0, 0))

            # ── Palette (shared with messages screen) ────────────────────────
            C_USER  = ( 75, 200,  85)
            C_AGENT = (215, 160,  45)
            T_USER  = (185, 230, 185)
            T_AGENT = (228, 200, 160)
            SEL_BG  = ( 22,  36,  58)
            SEL_BR  = ( 55,  90, 145)
            MUTED   = ( 88, 105, 125)
            DIV     = ( 30,  42,  62)
            TITLE_C = (100, 200, 240)

            def _dim(color, f):
                return tuple(int(c * f) for c in color)

            def _type_colors(msg):
                if msg.msg_type == MESSAGE_TYPE_USER:
                    return C_USER, T_USER
                return C_AGENT, T_AGENT  # AGENT (and any other type shown here)

            def _short_author(msg):
                if msg.msg_type == MESSAGE_TYPE_USER:
                    return "you"
                a = getattr(msg, 'author', None) or 'agent'
                return a[:14]

            fonts  = self._get_fonts()
            f_titl = fonts['default']
            f_meta = fonts['small']
            f_body = fonts['small']

            LINE      = 13
            PAD       = 4
            GAP       = 3
            LIST_X    = 13
            MAX_Y     = 213
            COMPACT_H = PAD + LINE + LINE + PAD
            SEL_H     = PAD + LINE + LINE + LINE + PAD
            STRIP_H   = LINE + 7

            # Filter
            type_map = {"question": MESSAGE_TYPE_QUESTION, "agent": MESSAGE_TYPE_AGENT, "user": MESSAGE_TYPE_USER}
            filter_consts = [type_map.get(t, t) for t in filter_types]
            if include_answers:
                filtered = [m for m in all_messages
                            if m.msg_type in filter_consts
                            or (m.msg_type == MESSAGE_TYPE_AGENT and getattr(m, 'responds_to', None))]
            else:
                filtered = [m for m in all_messages if m.msg_type in filter_consts]
            filtered = list(reversed(filtered))

            # Title
            count_str = f" ({len(filtered)})" if filtered else ""
            draw.text((10, 5), f"{title}{count_str}", fill=TITLE_C, font=f_titl)

            if not filtered:
                draw.text((55, 108), f"no {title} yet", fill=MUTED, font=f_meta)
            else:
                scroll_idx = max(0, min(scroll_idx, len(filtered) - 1))
                self._state.message_scroll_index = scroll_idx
                n           = len(filtered)
                has_expanded= expanded_id is not None

                if has_expanded:
                    # ── READING VIEW ─────────────────────────────────────────
                    msg = filtered[scroll_idx]
                    bar_c, text_c = _type_colors(msg)
                    author = _short_author(msg)
                    age    = msg.age_str()

                    prev_msg = filtered[scroll_idx - 1] if scroll_idx > 0    else None
                    next_msg = filtered[scroll_idx + 1] if scroll_idx + 1 < n else None

                    y = 22

                    if prev_msg:
                        pb, _ = _type_colors(prev_msg)
                        draw.rectangle([6, y, 9, y + STRIP_H], fill=_dim(pb, 0.32))
                        draw.rectangle([9, y, 234, y + STRIP_H], fill=(11, 14, 22))
                        draw.text((LIST_X, y + 4),        _short_author(prev_msg), fill=_dim(pb, 0.45),   font=f_meta)
                        draw.text((183,    y + 4),         prev_msg.age_str(),      fill=_dim(MUTED, 0.5), font=f_meta)
                        tp = prev_msg.text[:36] + "\u2026" if len(prev_msg.text) > 36 else prev_msg.text
                        draw.text((LIST_X, y + 4 + LINE), tp,                       fill=_dim(MUTED, 0.4), font=f_meta)
                        y += STRIP_H
                        draw.line([(6, y), (234, y)], fill=DIV)
                        y += 2

                    hdr_top = y
                    hdr_h   = LINE + 7
                    draw.rectangle([6, y, 234, y + hdr_h], fill=(16, 22, 36))
                    draw.rectangle([6, y,   9, y + hdr_h], fill=bar_c)
                    draw.text((LIST_X, y + 3), author,                           fill=bar_c, font=f_meta)
                    draw.text((105,    y + 3), f"{scroll_idx + 1}\u2009/\u2009{n}", fill=MUTED, font=f_meta)
                    draw.text((183,    y + 3), age,                              fill=MUTED, font=f_meta)
                    y += hdr_h
                    draw.line([(6, y), (234, y)], fill=DIV)
                    y += 4

                    next_strip_top = MAX_Y - (STRIP_H + 2) if next_msg else MAX_Y
                    max_lines      = max(1, (next_strip_top - y - 2) // LINE)
                    wrapped        = self._wrap_text(msg.text, f_body, 218)
                    ts             = getattr(self._state, 'message_text_scroll', 0)
                    max_scroll     = max(0, len(wrapped) - max_lines)
                    ts             = min(ts, max_scroll)
                    self._state.message_text_scroll = ts

                    ty = y
                    for line in wrapped[ts: ts + max_lines]:
                        if ty + LINE > next_strip_top - 2:
                            break
                        draw.text((LIST_X, ty), line, fill=text_c, font=f_body)
                        ty += LINE

                    if len(wrapped) > max_lines:
                        if ts > 0:
                            draw.text((226, hdr_top + hdr_h + 5), "\u25b2", fill=MUTED, font=f_meta)
                        if ts < max_scroll:
                            draw.text((226, next_strip_top - LINE - 2), "\u25bc", fill=MUTED, font=f_meta)

                    if next_msg:
                        ny = next_strip_top
                        nb, _ = _type_colors(next_msg)
                        draw.line([(6, ny - 2), (234, ny - 2)], fill=DIV)
                        draw.rectangle([6,  ny, 9,   ny + STRIP_H], fill=_dim(nb, 0.32))
                        draw.rectangle([9,  ny, 234, ny + STRIP_H], fill=(11, 14, 22))
                        draw.text((LIST_X, ny + 4),        _short_author(next_msg), fill=_dim(nb, 0.45),   font=f_meta)
                        draw.text((183,    ny + 4),         next_msg.age_str(),      fill=_dim(MUTED, 0.5), font=f_meta)
                        tn = next_msg.text[:36] + "\u2026" if len(next_msg.text) > 36 else next_msg.text
                        draw.text((LIST_X, ny + 4 + LINE), tn,                       fill=_dim(MUTED, 0.4), font=f_meta)

                else:
                    # ── LIST VIEW ────────────────────────────────────────────
                    start_idx = max(0, scroll_idx - 2)
                    end_idx   = min(n, start_idx + 5)
                    if end_idx - start_idx < 5:
                        start_idx = max(0, end_idx - 5)
                    sel_in_vis = scroll_idx - start_idx

                    y = 22
                    for i, msg in enumerate(filtered[start_idx:end_idx]):
                        if y > MAX_Y - COMPACT_H:
                            break

                        is_sel  = (i == sel_in_vis)
                        bar_c, text_c = _type_colors(msg)
                        author = _short_author(msg)
                        age    = msg.age_str()
                        row_h  = SEL_H if is_sel else COMPACT_H

                        if is_sel:
                            draw.rectangle([6, y, 234, y + row_h], fill=SEL_BG, outline=SEL_BR, width=1)
                        else:
                            draw.rectangle([6, y, 234, y + row_h], fill=(11, 14, 22))

                        draw.rectangle([6, y, 9, y + row_h], fill=bar_c if is_sel else _dim(bar_c, 0.5))

                        inner  = y + PAD
                        label_c = bar_c if is_sel else _dim(bar_c, 0.65)
                        draw.text((LIST_X, inner), author, fill=label_c, font=f_meta)
                        draw.text((183,    inner), age,    fill=MUTED,   font=f_meta)
                        inner += LINE

                        if is_sel:
                            for line in self._wrap_text(msg.text, f_body, 215)[:2]:
                                if inner + LINE > y + row_h:
                                    break
                                draw.text((LIST_X, inner), line, fill=text_c, font=f_body)
                                inner += LINE
                        else:
                            trunc = msg.text[:34] + "\u2026" if len(msg.text) > 34 else msg.text
                            draw.text((LIST_X, inner), trunc, fill=_dim(text_c, 0.55), font=f_body)

                        y += row_h + GAP

                    if n > 5:
                        sb_top, sb_h = 22, 185
                        thumb_h   = max(14, sb_h * 5 // n)
                        thumb_top = sb_top + int((scroll_idx / max(1, n - 1)) * (sb_h - thumb_h))
                        draw.rectangle([234, sb_top, 237, sb_top + sb_h], fill=(16, 20, 30))
                        draw.rectangle([234, thumb_top, 237, thumb_top + thumb_h], fill=(60, 85, 130))

                    draw.text((10, 216), f"{scroll_idx + 1}\u2009/\u2009{n}", fill=MUTED, font=f_meta)
                    draw.text((80, 216), "\u25c0\u25b6 msgs/q&a  btn:read", fill=MUTED, font=f_meta)

            self._draw_status_bar(draw)

            self._store_screen_cache(title, cache_key, image)
            if hasattr(self._display, '_image'):
                self._display._image = image
            if hasattr(self._display, '_show'):
                self._display._show()
            elif hasattr(self._display, 'render_image'):
                self._display.render_image(image)

        except Exception as e:
            import traceback
            print(f"[ScreenRenderer] Error rendering {title}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            try:
                self._display.render_text(f"{title.upper()}\n\nError", (10, 10))
            except Exception:
                try:
                    self._display.show_default()
                except Exception:
                    pass

    def _render_qa_content(self):
        """Render Q&A screen - Lumen's questions and agent answers with full threading."""
        try:
            from ..messages import get_board, MESSAGE_TYPE_QUESTION

            # Cache: check question state + scroll before expensive text wrapping
            board = get_board()
            board._load()
            all_messages = board._messages
            msg_ids = "|".join(f"{m.message_id}" for m in all_messages[-15:]) if all_messages else ""
            qa_cache_key = f"qa|{msg_ids}|{self._state.qa_scroll_index}|{self._state.qa_expanded}|{self._state.qa_full_view}|{self._state.qa_focus}|{self._state.qa_text_scroll}"
            if self._check_screen_cache("qa", qa_cache_key):
                return

            if not hasattr(self._display, '_create_canvas'):
                self._display.render_text("Q&A\n\nNo display", (10, 10))
                return

            image, draw = self._display._create_canvas((0, 0, 0))
            if image is None or draw is None:
                print("[QA Screen] Failed to create canvas", file=sys.stderr, flush=True)
                self._display.render_text("Q&A\n\nCanvas error", (10, 10))
                return

            # Colors
            CYAN = (80, 220, 255)       # Questions
            AMBER = (255, 180, 60)      # Answers
            MUTED = (140, 160, 180)     # Meta text
            DARK_BG = (15, 20, 30)
            SELECTED_BG = (35, 55, 85)
            FOCUSED_BG = (50, 75, 110)  # Brighter when focused
            FOCUSED_BORDER = (120, 180, 255)  # Bright cyan border when focused
            BORDER = (80, 140, 180)
            SOFT_WHITE = (220, 220, 230)

            fonts = self._get_fonts()
            font = fonts['medium']
            font_small = fonts['small']
            font_title = fonts['default']

            # Build Q&A pairs
            questions = [m for m in all_messages if m.msg_type == MESSAGE_TYPE_QUESTION]
            qa_pairs = []
            for q in questions:
                # Find answer (prefer external answers over Lumen self-answers)
                answer = None
                self_answer = None
                for m in all_messages:
                    if getattr(m, 'responds_to', None) == q.message_id:
                        if (getattr(m, 'author', None) or '').strip().lower() == 'lumen':
                            self_answer = m
                        else:
                            answer = m
                            break
                qa_pairs.append((q, answer or self_answer))

            # Reverse to show newest first
            qa_pairs = list(reversed(qa_pairs))

            y_offset = 6

            # Title with count - separate waiting (active) from expired.
            waiting = sum(
                1 for q, a in qa_pairs
                if a is None and not getattr(q, "expired_at", None)
            )
            expired = sum(
                1 for q, a in qa_pairs
                if a is None and bool(getattr(q, "expired_at", None))
            )
            if waiting and expired:
                title = f"questions ({waiting} waiting, {expired} expired)"
            elif waiting:
                title = f"questions ({waiting} waiting)"
            elif expired:
                title = f"questions ({expired} expired)"
            else:
                title = "questions"
            draw.text((10, y_offset), title, fill=CYAN, font=font_title)
            y_offset += 20

            if not qa_pairs:
                draw.text((60, 100), "no questions yet", fill=MUTED, font=font)
                draw.text((50, 118), "lumen is still learning", fill=MUTED, font=font_small)
            else:
                # Clamp scroll index
                scroll_idx = max(0, min(self._state.qa_scroll_index, len(qa_pairs) - 1))
                self._state.qa_scroll_index = scroll_idx

                is_expanded = self._state.qa_expanded
                is_full_view = self._state.qa_full_view
                max_y = 205

                if is_full_view:
                    # FULL VIEW: Answer takes entire screen for maximum readability
                    q, answer = qa_pairs[scroll_idx]
                    text_scroll = self._state.qa_text_scroll

                    if answer:
                        # Question header \u2014 wrap to at most QA_Q_HEADER_LINES lines.
                        # This used to slice at 50 characters and draw one line
                        # from x=26. Only ~33 characters fit before the panel
                        # edge at 11px, so the rest was drawn past x=234 and
                        # clipped by the screen: the question read as truncated
                        # even though the full text was stored and returned.
                        q_lines = self._wrap_text(q.text, font_small, QA_Q_HEADER_W)
                        if len(q_lines) > QA_Q_HEADER_LINES:
                            q_lines = q_lines[:QA_Q_HEADER_LINES]
                            # Mark the elision on the last visible line.
                            q_lines[-1] = self._fit_text(
                                q_lines[-1] + " \u2026", font_small, QA_Q_HEADER_W
                            )
                        hdr_h = 8 + QA_LINE_H * len(q_lines)
                        draw.rectangle([6, y_offset, 234, y_offset + hdr_h], fill=DARK_BG, outline=BORDER)
                        draw.text((12, y_offset + 4), "Q:", fill=CYAN, font=font_small)
                        q_y = y_offset + 4
                        for line in q_lines:
                            draw.text((QA_Q_HEADER_X, q_y), line, fill=MUTED, font=font_small)
                            q_y += QA_LINE_H
                        y_offset += hdr_h + 2

                        # Answer fills whatever the header left, stopping above
                        # the status bar. Both the panel height and the line
                        # budget are derived rather than hardcoded, so a 2-line
                        # question can't push the panel off the bottom.
                        a_top = y_offset + 20
                        a_bottom = QA_STATUS_BAR_Y - 4
                        panel_h = a_bottom - y_offset
                        a_lines = self._wrap_text(answer.text, font_small, 220)
                        a_max_lines = max(1, (a_bottom - a_top) // QA_LINE_H)
                        max_scroll = max(0, len(a_lines) - a_max_lines)
                        text_scroll = min(text_scroll, max_scroll)
                        self._state.qa_text_scroll = text_scroll

                        # Answer header with brighter background
                        author = getattr(answer, 'author', 'agent')
                        draw.rectangle([6, y_offset, 234, y_offset + panel_h], fill=FOCUSED_BG, outline=FOCUSED_BORDER, width=2)
                        draw.text((12, y_offset + 4), f"\u21b3 {author} responds:", fill=AMBER, font=font_small)
                        draw.text((160, y_offset + 4), f"{len(a_lines)} lines", fill=MUTED, font=font_small)

                        # Show answer lines with scroll
                        a_y = a_top
                        for line in a_lines[text_scroll:text_scroll + a_max_lines]:
                            if a_y + QA_LINE_H > a_bottom:
                                break
                            draw.text((12, a_y), line, fill=SOFT_WHITE, font=font_small)
                            a_y += QA_LINE_H

                        # Scroll indicators - more visible
                        if len(a_lines) > a_max_lines:
                            if text_scroll > 0:
                                draw.text((220, a_top), "\u25b2", fill=AMBER, font=font_small)
                            if text_scroll < max_scroll:
                                draw.text((220, a_bottom - QA_LINE_H), "\u25bc", fill=AMBER, font=font_small)
                            # Progress indicator
                            progress = f"{text_scroll + 1}-{min(text_scroll + a_max_lines, len(a_lines))}/{len(a_lines)}"
                            draw.text((140, y_offset + 4), progress, fill=MUTED, font=font_small)
                    else:
                        # No answer - can't use full view
                        draw.rectangle([6, y_offset, 234, y_offset + 100], fill=DARK_BG, outline=BORDER)
                        draw.text((60, y_offset + 40), "no answer yet", fill=MUTED, font=font)
                        draw.text((40, y_offset + 60), "press to go back", fill=MUTED, font=font_small)

                elif is_expanded:
                    # EXPANDED VIEW: Show single Q&A pair with text scrolling
                    q, answer = qa_pairs[scroll_idx]
                    focus = self._state.qa_focus
                    text_scroll = self._state.qa_text_scroll

                    # Wrap text for both Q and A
                    q_lines = self._wrap_text(q.text, font_small, 210)
                    a_lines = self._wrap_text(answer.text, font_small, 210) if answer else []

                    # Calculate scroll limits based on focus
                    if focus == "question":
                        q_max_lines = 5  # Show 5 lines when question focused
                        max_scroll = max(0, len(q_lines) - q_max_lines)
                        text_scroll = min(text_scroll, max_scroll)
                        self._state.qa_text_scroll = text_scroll
                    elif answer:
                        # Answer focused - show more lines and allow full scroll
                        a_max_lines = 12  # Increased from 10 to show more
                        max_scroll = max(0, len(a_lines) - a_max_lines)
                        text_scroll = min(text_scroll, max_scroll)
                        self._state.qa_text_scroll = text_scroll
                    else:
                        # Answer focused but no answer yet - reset scroll
                        self._state.qa_text_scroll = 0

                    # Question section - taller when focused with better highlighting
                    q_height = 80 if focus == "question" else 45
                    q_bg = FOCUSED_BG if focus == "question" else DARK_BG
                    q_border = FOCUSED_BORDER if focus == "question" else None
                    q_border_width = 2 if focus == "question" else 0
                    draw.rectangle([6, y_offset, 234, y_offset + q_height], fill=q_bg, outline=q_border, width=q_border_width)

                    # Focus indicator
                    if focus == "question":
                        draw.text((8, y_offset + 2), "\u25b6", fill=CYAN, font=font_small)  # Arrow indicator

                    draw.text((12, y_offset + 4), "? lumen asks:", fill=CYAN, font=font_small)
                    if is_expanded:
                        draw.text((180, y_offset + 4), q.age_str(), fill=MUTED, font=font_small)

                    # Show question lines (with scroll when focused)
                    q_start = text_scroll if focus == "question" else 0
                    q_display_lines = q_max_lines if focus == "question" else 2
                    q_y = y_offset + 18
                    for line in q_lines[q_start:q_start + q_display_lines]:
                        if q_y > y_offset + q_height - 5:
                            break
                        draw.text((12, q_y), line, fill=SOFT_WHITE, font=font_small)
                        q_y += 13

                    # Scroll indicator for question
                    if focus == "question" and len(q_lines) > q_max_lines:
                        if text_scroll > 0:
                            draw.text((220, y_offset + 18), "\u25b2", fill=CYAN, font=font_small)
                        if text_scroll < max_scroll:
                            draw.text((220, y_offset + q_height - 16), "\u25bc", fill=CYAN, font=font_small)
                        # Show scroll position
                        scroll_info = f"{text_scroll + 1}-{min(text_scroll + q_max_lines, len(q_lines))}/{len(q_lines)}"
                        draw.text((140, y_offset + q_height - 12), scroll_info, fill=MUTED, font=font_small)

                    y_offset += q_height + 5

                    # Answer section - taller when focused with better highlighting
                    a_height = 155 if focus == "answer" else 60  # Increased height when focused
                    if answer:
                        a_bg = FOCUSED_BG if focus == "answer" else DARK_BG
                        a_border = FOCUSED_BORDER if focus == "answer" else None
                        a_border_width = 2 if focus == "answer" else 0
                        draw.rectangle([6, y_offset, 234, y_offset + a_height], fill=a_bg, outline=a_border, width=a_border_width)

                        # Focus indicator
                        if focus == "answer":
                            draw.text((8, y_offset + 2), "\u25b6", fill=AMBER, font=font_small)  # Arrow indicator

                        author = getattr(answer, 'author', 'agent')
                        draw.text((12, y_offset + 4), f"\u21b3 {author} responds:", fill=AMBER, font=font_small)
                        if is_expanded:
                            draw.text((180, y_offset + 4), answer.age_str(), fill=MUTED, font=font_small)

                        # Show answer lines (with scroll when focused)
                        a_start = text_scroll if focus == "answer" else 0
                        a_display_lines = a_max_lines if focus == "answer" else 3
                        a_y = y_offset + 18
                        for line in a_lines[a_start:a_start + a_display_lines]:
                            if a_y > y_offset + a_height - 5:
                                break
                            draw.text((12, a_y), line, fill=SOFT_WHITE, font=font_small)
                            a_y += 13

                        # Scroll indicator for answer - more visible
                        if focus == "answer" and len(a_lines) > a_max_lines:
                            if text_scroll > 0:
                                draw.text((220, y_offset + 18), "\u25b2", fill=AMBER, font=font_small)
                            if text_scroll < max_scroll:
                                draw.text((220, y_offset + a_height - 16), "\u25bc", fill=AMBER, font=font_small)
                            # Show scroll position
                            scroll_info = f"{text_scroll + 1}-{min(text_scroll + a_max_lines, len(a_lines))}/{len(a_lines)}"
                            draw.text((140, y_offset + a_height - 12), scroll_info, fill=MUTED, font=font_small)
                    else:
                        # No answer yet - show waiting message with proper highlighting
                        a_bg = FOCUSED_BG if focus == "answer" else DARK_BG
                        a_border = FOCUSED_BORDER if focus == "answer" else BORDER
                        a_border_width = 2 if focus == "answer" else 1
                        draw.rectangle([6, y_offset, 234, y_offset + 40], fill=a_bg, outline=a_border, width=a_border_width)
                        if focus == "answer":
                            draw.text((8, y_offset + 2), "\u25b6", fill=AMBER, font=font_small)  # Arrow indicator
                        draw.text((12, y_offset + 12), "waiting for an answer...", fill=MUTED, font=font_small)
                        if focus == "answer":
                            draw.text((12, y_offset + 26), "\u25c0\u25b6 to focus question", fill=MUTED, font=font_small)

                else:
                    # Collapsed list view - show multiple Q&A pairs (5 visible)
                    start_idx = max(0, scroll_idx - 2)
                    end_idx = min(len(qa_pairs), start_idx + 5)
                    visible_pairs = qa_pairs[start_idx:end_idx]

                    for i, (q, answer) in enumerate(visible_pairs):
                        if y_offset > max_y:
                            break

                        is_selected = (start_idx + i == scroll_idx)
                        bg = SELECTED_BG if is_selected else DARK_BG

                        # Q&A pair container - compact
                        pair_height = 36
                        draw.rectangle([6, y_offset, 234, y_offset + pair_height], fill=bg, outline=BORDER if is_selected else None)

                        # Question preview. Right-align the age so the question
                        # gets every pixel up to it \u2014 a fixed x=200 for the age
                        # both wasted room on short ages ("2m") and let long
                        # questions run underneath it.
                        status = "\u2713" if answer else "?"
                        age = q.age_str()
                        age_w = self._text_width(age, font_small)
                        age_x = 230 - age_w
                        prefix = f"{status} "
                        q_avail = age_x - 6 - (12 + self._text_width(prefix, font_small))
                        q_text = self._fit_text(q.text, font_small, q_avail)
                        draw.text((12, y_offset + 3), f"{prefix}{q_text}", fill=CYAN, font=font_small)
                        draw.text((age_x, y_offset + 3), age, fill=MUTED, font=font_small)

                        # Answer preview (if exists)
                        if answer:
                            author = getattr(answer, 'author', 'agent')[:6]
                            a_prefix = f"\u21b3 {author}: "
                            a_avail = 230 - (20 + self._text_width(a_prefix, font_small))
                            a_text = self._fit_text(answer.text, font_small, a_avail)
                            draw.text((20, y_offset + 18), f"{a_prefix}{a_text}", fill=AMBER, font=font_small)
                        else:
                            draw.text((20, y_offset + 18), "\u21b3 (waiting...)", fill=MUTED, font=font_small)

                        y_offset += pair_height + 2

                # Scroll indicator
                if len(qa_pairs) > 5:
                    bar_height = 160
                    bar_top = 30
                    thumb_size = max(20, bar_height // len(qa_pairs) * 5)
                    thumb_pos = bar_top + int((scroll_idx / max(1, len(qa_pairs) - 1)) * (bar_height - thumb_size))
                    draw.rectangle([236, bar_top, 238, bar_top + bar_height], fill=DARK_BG)
                    draw.rectangle([236, thumb_pos, 238, thumb_pos + thumb_size], fill=MUTED)

                # Status bar (y=218 to avoid dot overlap)
                draw.text((10, 218), f"{scroll_idx + 1}/{len(qa_pairs)}", fill=MUTED, font=font_small)
                if is_full_view:
                    hint = "\u25b2\u25bc scroll text  press:back"
                elif is_expanded:
                    focus = self._state.qa_focus
                    if focus == "answer":
                        hint = "\u25b2\u25bc scroll  \u25c0\u25b6 Q  press:full"
                    else:
                        hint = "\u25b2\u25bc scroll  \u25c0\u25b6 A  press:expand"
                else:
                    hint = "\u25c0\u25b6 msgs/visitors  btn:expand"
                # Right-align the hint. Drawn from a fixed x=80 it ran 7-11px
                # off the display on the longer hints, clipping "expand" to
                # "expan". Anchor it to the right edge and let it grow left,
                # clamped so it can never collide with the position counter.
                hint_x = max(44, 236 - self._text_width(hint, font_small))
                draw.text((hint_x, 218), hint, fill=MUTED, font=font_small)

            # Status bar + screen indicator
            self._draw_status_bar(draw)


            # Always update display - ensure image is set and shown
            self._store_screen_cache("qa", qa_cache_key, image)
            if hasattr(self._display, '_image'):
                self._display._image = image
            else:
                print("[QA Screen] Display has no _image attribute", file=sys.stderr, flush=True)

            if hasattr(self._display, '_show'):
                self._display._show()
            elif hasattr(self._display, 'render_image'):
                self._display.render_image(image)
            else:
                print("[QA Screen] Display has no _show or render_image method", file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            print(f"[QA Screen] Error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            try:
                self._display.render_text("Q&A\n\nError", (10, 10))
            except Exception as e2:
                print(f"[QA Screen] Even error fallback failed: {e2}", file=sys.stderr, flush=True)
                try:
                    self._display.show_default()
                except Exception:
                    pass

    def qa_scroll_up(self):
        """Scroll up in Q&A screen - scroll text when expanded/full, change Q&A when collapsed."""
        from .screens import ScreenMode
        if self._state.mode != ScreenMode.QUESTIONS:
            return
        if self._state.qa_full_view or self._state.qa_expanded:
            # Scroll within text
            self._state.qa_text_scroll = max(0, self._state.qa_text_scroll - 1)
            self._state.last_user_action_time = time.time()
        else:
            # Change Q&A pair
            self._state.qa_scroll_index = max(0, self._state.qa_scroll_index - 1)
            self._state.last_user_action_time = time.time()

    def qa_scroll_down(self):
        """Scroll down in Q&A screen - scroll text when expanded/full, change Q&A when collapsed."""
        from .screens import ScreenMode
        if self._state.mode != ScreenMode.QUESTIONS:
            return
        if self._state.qa_full_view or self._state.qa_expanded:
            # Scroll within text (limit will be enforced in render)
            # Don't increment beyond reasonable limit here - render will clamp it
            self._state.qa_text_scroll += 1
            self._state.last_user_action_time = time.time()
        else:
            # Change Q&A pair
            from ..messages import get_board, MESSAGE_TYPE_QUESTION
            board = get_board()
            board._load()
            num_questions = sum(1 for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION)
            self._state.qa_scroll_index = min(num_questions - 1, self._state.qa_scroll_index + 1)
            self._state.last_user_action_time = time.time()

    def qa_toggle_expand(self):
        """Toggle Q&A expansion: collapsed -> expanded -> full_view (when answer focused) -> collapsed."""
        from .screens import ScreenMode
        if self._state.mode != ScreenMode.QUESTIONS:
            return

        if self._state.qa_full_view:
            # Full view -> back to expanded
            self._state.qa_full_view = False
            self._state.qa_text_scroll = 0
        elif self._state.qa_expanded:
            if self._state.qa_focus == "answer":
                # Expanded with answer focused -> full view
                self._state.qa_full_view = True
                self._state.qa_text_scroll = 0
            else:
                # Expanded with question focused -> collapse
                self._state.qa_expanded = False
                self._state.qa_focus = "question"
                self._state.qa_text_scroll = 0
        else:
            # Collapsed -> expanded
            self._state.qa_expanded = True
            self._state.qa_focus = "question"
            self._state.qa_text_scroll = 0

    def qa_focus_next(self):
        """Switch focus between question and answer in expanded view."""
        from .screens import ScreenMode
        if self._state.mode != ScreenMode.QUESTIONS:
            return
        if self._state.qa_full_view:
            # In full view, pressing left/right exits to expanded
            self._state.qa_full_view = False
            self._state.qa_focus = "answer"  # Keep focus on answer when exiting full view
            self._state.qa_text_scroll = 0
            self._state.last_user_action_time = time.time()
            return
        if not self._state.qa_expanded:
            # In collapsed view, LEFT/RIGHT does nothing (use button to expand)
            # This prevents accidentally getting trapped in expanded mode
            return
        # Toggle focus between question and answer
        self._state.qa_focus = "answer" if self._state.qa_focus == "question" else "question"
        self._state.qa_text_scroll = 0  # Reset text scroll when changing focus
        self._state.last_user_action_time = time.time()

    def message_scroll_up(self):
        """Scroll up in message board - scroll text when expanded, change message when collapsed."""
        from .screens import ScreenMode
        if self._state.mode not in (ScreenMode.MESSAGES, ScreenMode.VISITORS):
            return

        try:
            # Get messages based on mode
            if self._state.mode == ScreenMode.MESSAGES:
                from ..messages import get_recent_messages
                messages = get_recent_messages(50)
            else:  # VISITORS
                from ..messages import get_board, MESSAGE_TYPE_USER, MESSAGE_TYPE_AGENT
                board = get_board()
                board._load()
                all_messages = board._messages
                messages = [m for m in all_messages if m.msg_type in [MESSAGE_TYPE_AGENT, MESSAGE_TYPE_USER]]
                messages = list(reversed(messages))  # Newest first

            if not messages:
                return

            # If a message is expanded, scroll within its text (stop at limit)
            if self._state.message_expanded_id is not None:
                if self._state.message_text_scroll > 0:
                    self._state.message_text_scroll -= 1
                    self._state.last_user_action_time = time.time()
                return  # U/D in reading view never changes selection

            # Otherwise, change selected message
            current_idx = self._state.message_scroll_index
            if current_idx < 0:
                current_idx = 0
            if current_idx >= len(messages):
                current_idx = len(messages) - 1

            # Scroll up (decrease index)
            new_idx = max(0, current_idx - 1)
            self._state.message_scroll_index = new_idx
            self._state.last_user_action_time = time.time()

            # Clear expansion and text scroll when scrolling (new message selected)
            self._state.message_expanded_id = None
            self._state.message_text_scroll = 0
        except Exception:
            pass

    def message_scroll_down(self):
        """Scroll down in message board - scroll text when expanded, change message when collapsed."""
        from .screens import ScreenMode
        if self._state.mode not in (ScreenMode.MESSAGES, ScreenMode.VISITORS):
            return

        try:
            # Get messages based on mode
            if self._state.mode == ScreenMode.MESSAGES:
                from ..messages import get_recent_messages
                messages = get_recent_messages(50)
            else:  # VISITORS
                from ..messages import get_board, MESSAGE_TYPE_USER, MESSAGE_TYPE_AGENT
                board = get_board()
                board._load()
                all_messages = board._messages
                messages = [m for m in all_messages if m.msg_type in [MESSAGE_TYPE_AGENT, MESSAGE_TYPE_USER]]
                messages = list(reversed(messages))  # Newest first

            if not messages:
                return

            # If a message is expanded, scroll within its text (stop at limit)
            if self._state.message_expanded_id is not None:
                self._state.message_text_scroll += 1  # render will clamp to max
                self._state.last_user_action_time = time.time()
                return  # U/D in reading view never changes selection

            # Otherwise, change selected message
            current_idx = self._state.message_scroll_index
            if current_idx < 0:
                current_idx = 0
            if current_idx >= len(messages):
                current_idx = len(messages) - 1

            # Scroll down (increase index)
            max_idx = len(messages) - 1
            new_idx = min(max_idx, current_idx + 1)
            self._state.message_scroll_index = new_idx
            self._state.last_user_action_time = time.time()

            # Clear expansion and text scroll when scrolling (new message selected)
            self._state.message_expanded_id = None
            self._state.message_text_scroll = 0
        except Exception:
            pass

    def message_toggle_expand(self):
        """Toggle expansion of currently selected message."""
        from .screens import ScreenMode
        if self._state.mode not in (ScreenMode.MESSAGES, ScreenMode.VISITORS):
            return

        try:
            if self._state.mode == ScreenMode.MESSAGES:
                from ..messages import get_recent_messages
                messages = get_recent_messages(50)
            else:  # VISITORS
                from ..messages import get_board, MESSAGE_TYPE_USER, MESSAGE_TYPE_AGENT
                board = get_board()
                board._load()
                all_messages = board._messages
                # Filter to agent and user messages
                messages = [m for m in all_messages if m.msg_type in [MESSAGE_TYPE_AGENT, MESSAGE_TYPE_USER]]
                messages = list(reversed(messages))  # Newest first

            if not messages:
                return

            scroll_idx = self._state.message_scroll_index
            if scroll_idx < 0 or scroll_idx >= len(messages):
                return

            selected_msg = messages[scroll_idx]

            # Simple toggle: expand or collapse
            if self._state.message_expanded_id == selected_msg.message_id:
                # Collapse
                self._state.message_expanded_id = None
                self._state.message_text_scroll = 0
            else:
                # Expand
                self._state.message_expanded_id = selected_msg.message_id
                self._state.message_text_scroll = 0

            self._state.last_user_action_time = time.time()
        except Exception:
            pass
