import { createContext, useMemo } from 'react';
import createTheme from '@mui/material/styles/createTheme';

export type TModeType = 'light' | 'dark';

export const tokens = (mode: TModeType) => ({
    ...(mode === 'dark'
        ? {
              grey: {
                  100: '#B0B8C0',
                  200: '#9098A1',
                  300: '#707880',
                  400: '#505860',
                  500: '#303840',
                  600: '#202830',
                  700: '#101820',
                  800: '#0A1015',
                  900: '#05080A'
              },
              primary: {
                  100: '#0A1320',
                  200: '#08101C',
                  300: '#060C18',
                  400: '#040A15'
              },
              blue: {
                  100: '#6A7B8C',
                  200: '#3A4A5C',
                  300: '#1A2A3C'
              },
              text: {
                  primary: '#DDE1E6'
              }
          }
        : {
              grey: {
                  100: '#F0F4F8',
                  200: '#D8DFE5',
                  300: '#C0C9D0',
                  400: '#A8B1BB',
                  500: '#9099A6',
                  600: '#707880',
                  700: '#505860',
                  800: '#303840',
                  900: '#101820'
              },
              primary: {
                  100: '#B8C0C8',
                  200: '#90A0B0',
                  300: '#607080',
                  400: '#405060'
              },
              blue: {
                  100: '#6A7B8C',
                  200: '#3A4A5C',
                  300: '#1A2A3C'
              },
              text: {
                  primary: '#1A1F26'
              }
          })
});

export const themeSettings = (mode: TModeType) => {
    const colors = tokens(mode);

    return {
        palette: {
            mode: mode,
            ...(mode === 'dark'
                ? {
                      primary: {
                          lighter: colors.primary[100],
                          light: colors.primary[300],
                          main: colors.primary[300],
                          dark: colors.primary[500],
                      },
                      secondary: {
                          light: colors.blue[100],
                          main: colors.blue[200],
                          dark: colors.blue[300]
                      },
                      neutral: {
                          dark: colors.grey[700],
                          main: colors.grey[500],
                          light: colors.grey[100]
                      },
                      background: {
                          default:
                              'linear-gradient(0deg, rgb(50, 80, 120) 10%, rgb(100, 120, 160) 100%)'
                      }
                  }
                : {
                      primary: {
                          lighter: colors.primary[400],
                          light: colors.primary[300],
                          main: colors.primary[200],
                          dark: colors.primary[100]
                      },
                      secondary: {
                          light: colors.blue[100],
                          main: colors.blue[200],
                          dark: colors.blue[300]
                      },
                      neutral: {
                          dark: colors.grey[700],
                          main: colors.grey[500],
                          light: colors.grey[100]
                      },
                      background: {
                          default: colors.primary[100]
                      }
                  })
        },
        typography: {
            fontFamily: ['Inter', 'sans-serif'].join(','),
            fontSize: 12,
            h1: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 36
            },
            h2: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 28
            },
            h3: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 22
            },
            h4: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 18
            },
            h5: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 14
            },
            h6: {
                fontFamily: ['Inter', 'sans-serif'].join(','),
                fontSize: 12
            }
        },
        components: {
            MuiButton: {
                styleOverrides: {
                    root: {
                        fontSize: '12px',
                        padding: '4px 8px'
                    }
                }
            }
        }
    };
};

export const ColorModeContext = createContext({
    toggleColorMode: () => {}
});

export const useMode = (mode: TModeType) => {
    return useMemo(() => createTheme(themeSettings(mode)), [mode]);
};
