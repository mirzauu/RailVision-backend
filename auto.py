import pyautogui
import time
import argparse
import random
try:
	import msvcrt
except Exception:
	msvcrt = None


def run_mouse_mover(total_hours: float = 1.0,
					move_duration: float = 1.0,
					cycle_interval_min: float = 2.0,
					cycle_interval_max: float = 5.0,
					min_scroll: int = 50,
					max_scroll: int = 200,
					smooth_steps: int = 10,
					dry_run: bool = False,
					enable_keys: bool = True,
					key_min_interval: float = 30.0,
					key_max_interval: float = 120.0,
					key_min_presses: int = 1,
					key_max_presses: int = 3,
					key_list: list | None = None,
					enable_mouse_move: bool = True,
					mouse_move_min_interval: float = 10.0,
					mouse_move_max_interval: float = 60.0,
					enable_hotkey: bool = True,
					hotkey_interval: float = 420.0,
					hotkey: str | None = None):
	"""Simulate natural user activity with randomized scrolling, mouse moves, and key presses.

	- total_hours: how many hours to run (default 1.0)
	- move_duration: approximate seconds to spend performing the scroll action (default 1.0)
	- cycle_interval_min/max: random range for seconds between starts of consecutive scroll moves
	- min_scroll/max_scroll: range of scroll 'clicks' to perform each move; sign chosen randomly
	- smooth_steps: number of small scroll steps to perform during move_duration (simulates smooth scroll)
	- dry_run: when True, don't perform actual actions (useful for testing)
	- enable_mouse_move: enable random mouse pointer movements
	- mouse_move_min/max_interval: random range for seconds between mouse movements
	- hotkey_interval: seconds between ctrl+tab presses (default: 420 = 7 minutes)
	"""
	total_seconds = total_hours * 3600

	if cycle_interval_min > cycle_interval_max:
		cycle_interval_min, cycle_interval_max = cycle_interval_max, cycle_interval_min
	if min_scroll > max_scroll:
		min_scroll, max_scroll = max_scroll, min_scroll

	start = time.time()
	i = 0
	
	# Key press scheduling
	if key_list is None:
		# safe default key list: letters, digits, and some navigation keys
		key_list = list('abcdefghijklmnopqrstuvwxyz0123456789') + ['space', 'enter', 'tab', 'up', 'down', 'left', 'right', 'pageup', 'pagedown', 'home', 'end', 'esc']

	next_key_time = None
	if enable_keys:
		next_key_time = start + random.uniform(key_min_interval, key_max_interval)

	# Mouse movement scheduling
	next_mouse_time = None
	if enable_mouse_move:
		next_mouse_time = start + random.uniform(mouse_move_min_interval, mouse_move_max_interval)

	# Hotkey scheduling (ctrl+tab every 7 minutes)
	next_hotkey_time = None
	hotkey_keys = None
	if enable_hotkey:
		next_hotkey_time = start + hotkey_interval
		if hotkey:
			hotkey_keys = [k.strip() for k in hotkey.split("+") if k.strip()]
		else:
			# default combo: ctrl+tab
			hotkey_keys = ["ctrl", "tab"]

	print(f"Starting natural activity simulator for {total_hours} hour(s).")
	print(f"Scroll: interval=[{cycle_interval_min},{cycle_interval_max}]s, amount=[{min_scroll},{max_scroll}] clicks")
	if enable_keys:
		print(f"Keys: interval=[{key_min_interval},{key_max_interval}]s, presses=[{key_min_presses},{key_max_presses}]")
	if enable_mouse_move:
		print(f"Mouse: interval=[{mouse_move_min_interval},{mouse_move_max_interval}]s")
	if enable_hotkey:
		print(f"Hotkey: {'+'.join(hotkey_keys)} every {hotkey_interval}s (~{hotkey_interval/60:.1f} min)")
	print("Note: move mouse to top-left corner to trigger PyAutoGUI failsafe.")
	
	try:
		while time.time() - start < total_seconds:
			i += 1
			now = time.strftime("%H:%M:%S")

			# Randomize scroll amount and direction
			amount = random.randint(min_scroll, max_scroll)
			direction = random.choice([1, -1])
			total_clicks = amount * direction

			print(f"[{i}] {now} - scrolling {'up' if direction>0 else 'down'} by {amount} clicks over ~{move_duration}s")

			if dry_run:
				print("(dry-run) would scroll", total_clicks)
			else:
				# Smoothly perform the scroll in small steps over move_duration
				steps = max(1, int(smooth_steps))
				step_clicks = total_clicks // steps
				remainder = total_clicks - (step_clicks * steps)
				step_delay = max(0.0, move_duration / steps) if move_duration > 0 else 0

				for s in range(steps):
					sc = step_clicks + (1 if s == 0 and remainder != 0 else 0)
					if sc != 0:
						pyautogui.scroll(int(sc))
					if step_delay > 0:
						time.sleep(step_delay)

			# Check if it's time to do a random key press event
			if enable_keys and next_key_time is not None and time.time() >= next_key_time:
				kp_count = random.randint(max(1, key_min_presses), max(1, key_max_presses))
				now2 = time.strftime("%H:%M:%S")
				print(f"[{i}] {now2} - performing {kp_count} random key press(es)")

				if dry_run:
					print("(dry-run) would press keys:", ",".join(random.choice(key_list) for _ in range(kp_count)))
				else:
					pressed = []
					for _ in range(kp_count):
						k = random.choice(key_list)
						pressed.append(k)
						try:
							pyautogui.press(k)
							print(f"Pressed: {k}")
						except Exception:
							print(f"Warning: could not press key '{k}'")
						time.sleep(random.uniform(0.05, 0.25))
					if pressed:
						print("Pressed keys:", ",".join(pressed))

				# Schedule next key event with random interval
				next_key_time = time.time() + random.uniform(key_min_interval, key_max_interval)

			# Check if it's time to move mouse pointer
			if enable_mouse_move and next_mouse_time is not None and time.time() >= next_mouse_time:
				now3 = time.strftime("%H:%M:%S")
				# Get screen size
				screen_width, screen_height = pyautogui.size()
				# Random position (avoid edges and top-left corner for failsafe)
				x = random.randint(100, screen_width - 100)
				y = random.randint(100, screen_height - 100)
				duration = random.uniform(0.5, 2.0)
				
				print(f"[{i}] {now3} - moving mouse to ({x}, {y}) over {duration:.1f}s")
				
				if dry_run:
					print(f"(dry-run) would move mouse to ({x}, {y})")
				else:
					try:
						pyautogui.moveTo(x, y, duration=duration)
						print(f"Mouse moved to ({x}, {y})")
					except Exception as e:
						print(f"Warning: could not move mouse - {e}")
				
				# Schedule next mouse move with random interval
				next_mouse_time = time.time() + random.uniform(mouse_move_min_interval, mouse_move_max_interval)

			# Hotkey check (fixed interval - every 7 minutes)
			if enable_hotkey and next_hotkey_time is not None and time.time() >= next_hotkey_time:
				now4 = time.strftime("%H:%M:%S")
				print(f"[{i}] {now4} - performing hotkey {'+'.join(hotkey_keys)}")
				if dry_run:
					print(f"(dry-run) would press hotkey: {'+'.join(hotkey_keys)}")
				else:
					try:
						pyautogui.hotkey(*hotkey_keys)
						print(f"Hotkey pressed: {'+'.join(hotkey_keys)}")
					except Exception as e:
						print(f"Warning: could not press hotkey {'+'.join(hotkey_keys)} - {e}")
				# schedule next (fixed interval)
				next_hotkey_time = time.time() + hotkey_interval

			# Choose next scroll interval randomly between min and max
			next_interval = random.uniform(cycle_interval_min, cycle_interval_max)
			sleep_time = max(0.0, next_interval - move_duration)

			if sleep_time > 0:
				time.sleep(sleep_time)

			if i % 30 == 0:
				elapsed = int(time.time() - start)
				print(f"Elapsed {elapsed} seconds (~{elapsed/60:.1f} minutes)")

		print("Scheduled duration complete.")
	except KeyboardInterrupt:
		print("Interrupted by user (Ctrl+C). Exiting early.")
	except Exception as e:
		print("Error:", e)
	finally:
		print("Done.")


def parse_args():
	p = argparse.ArgumentParser(description="Natural activity simulator: perform repeated moves for a given time")
	p.add_argument("--hours", type=float, default=1.0, help="Total hours to run (default: 1)")
	p.add_argument("--move-duration", type=float, default=1.0, help="Seconds each scroll action should take (default: 1)")
	p.add_argument("--min-interval", type=float, default=2.0, help="Min seconds between scroll cycles (default: 2)")
	p.add_argument("--max-interval", type=float, default=5.0, help="Max seconds between scroll cycles (default: 5)")
	p.add_argument("--min-scroll", type=int, default=50, help="Min scroll 'clicks' per move (default: 50)")
	p.add_argument("--max-scroll", type=int, default=200, help="Max scroll 'clicks' per move (default: 200)")
	p.add_argument("--smooth-steps", type=int, default=10, help="Number of small steps to split each scroll into (default: 10)")
	p.add_argument("--dry-run", action="store_true", help="Don't perform real actions; just print what would happen")
	p.add_argument("--enable-keys", action="store_true", help="Enable random key presses during the run")
	p.add_argument("--key-min-interval", type=float, default=30.0, help="Min seconds between random key events (default: 30)")
	p.add_argument("--key-max-interval", type=float, default=120.0, help="Max seconds between random key events (default: 120)")
	p.add_argument("--key-min-presses", type=int, default=1, help="Min number of key presses per event (default: 1)")
	p.add_argument("--key-max-presses", type=int, default=3, help="Max number of key presses per event (default: 3)")
	p.add_argument("--key-list", type=str, default=None, help="Comma-separated list of keys to pick from (default: letters/digits/nav keys)")
	p.add_argument("--enable-mouse-move", action="store_true", help="Enable random mouse pointer movements")
	p.add_argument("--mouse-move-min-interval", type=float, default=10.0, help="Min seconds between mouse movements (default: 10)")
	p.add_argument("--mouse-move-max-interval", type=float, default=60.0, help="Max seconds between mouse movements (default: 60)")
	p.add_argument("--enable-hotkey", action="store_true", help="Enable periodic ctrl+tab presses (every 7 minutes)")
	p.add_argument("--hotkey-interval", type=float, default=420.0, help="Seconds between hotkey presses (default: 420 = 7 min)")
	p.add_argument("--hotkey", type=str, default=None, help="Hotkey combo to press, keys separated by + (default: ctrl+tab)")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	# Quick safety note: move your mouse to the top-left corner to trigger PyAutoGUI's failsafe
	key_list = None
	if args.key_list:
		key_list = [k.strip() for k in args.key_list.split(",") if k.strip()]

	run_mouse_mover(total_hours=args.hours,
					move_duration=args.move_duration,
					cycle_interval_min=args.min_interval,
					cycle_interval_max=args.max_interval,
					min_scroll=args.min_scroll,
					max_scroll=args.max_scroll,
					smooth_steps=args.smooth_steps,
					dry_run=args.dry_run,
					enable_keys=args.enable_keys,
					key_min_interval=args.key_min_interval,
					key_max_interval=args.key_max_interval,
					key_min_presses=args.key_min_presses,
					key_max_presses=args.key_max_presses,
					key_list=key_list,
					enable_mouse_move=args.enable_mouse_move,
					mouse_move_min_interval=args.mouse_move_min_interval,
					mouse_move_max_interval=args.mouse_move_max_interval,
					enable_hotkey=args.enable_hotkey,
					hotkey_interval=args.hotkey_interval,
					hotkey=args.hotkey)