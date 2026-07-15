use rusanta_viz::style::{Color, Style};

/// Statistical visualization theme.
///
/// A theme defines:
/// - default colors
/// - grid visibility
/// - background
/// - font scaling
#[derive(Debug, Clone)]
pub struct StatTheme {
    pub background: Color,
    pub grid: bool,
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
}

impl StatTheme {
    /// Convert into a rusanta-viz Style.
    pub fn to_style(&self) -> Style {
        let mut style = Style::default();

        style.background = self.background;
        style.grid = self.grid;
        style.primary = self.primary;
        style.secondary = self.secondary;
        style.accent = self.accent;

        style
    }
}

/// Light theme (default).
///
/// Similar to seaborn whitegrid.
pub fn light() -> StatTheme {
    StatTheme {
        background: Color::WHITE,
        grid: true,
        primary: Color::BLUE,
        secondary: Color::GRAY,
        accent: Color::RED,
    }
}

/// Dark theme.
///
/// Suitable for dashboards.
pub fn dark() -> StatTheme {
    StatTheme {
        background: Color::rgb(0.1, 0.1, 0.1),
        grid: true,
        primary: Color::CYAN,
        secondary: Color::rgb(0.6, 0.6, 0.6),
        accent: Color::ORANGE,
    }
}

/// Minimal theme.
///
/// No grid, muted colors.
pub fn minimal() -> StatTheme {
    StatTheme {
        background: Color::WHITE,
        grid: false,
        primary: Color::BLACK,
        secondary: Color::GRAY,
        accent: Color::BLUE,
    }
}

/// Scientific publication theme.
///
/// High contrast, no noise.
pub fn publication() -> StatTheme {
    StatTheme {
        background: Color::WHITE,
        grid: false,
        primary: Color::BLACK,
        secondary: Color::rgb(0.3, 0.3, 0.3),
        accent: Color::RED,
    }
}
